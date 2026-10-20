#ifndef __AM_AST_H__
#define __AM_AST_H__

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "am_object.h"
#include "am_lexer.h"
#include "am_allocator.h"
#include "am_vocab.h"
#include "am_heap.h"
#include "am_map.h"
#include "am_list.h"
#include "am_wstring.h"


// VM指令集（操作码）枚举

#define AM_VM_OP_nop         (0)
#define AM_VM_OP_store       (1)
#define AM_VM_OP_load        (2)
#define AM_VM_OP_loadclosure (3)
#define AM_VM_OP_push        (4)
#define AM_VM_OP_pop         (5)
#define AM_VM_OP_swap        (6)
#define AM_VM_OP_set         (7)
#define AM_VM_OP_call        (8)
#define AM_VM_OP_callnative  (9)
#define AM_VM_OP_tailcall    (10)
#define AM_VM_OP_return      (11)
#define AM_VM_OP_capturecc   (12)
#define AM_VM_OP_iftrue      (13)
#define AM_VM_OP_iffalse     (14)
#define AM_VM_OP_goto        (15)
#define AM_VM_OP_read        (16)
#define AM_VM_OP_write       (17)
#define AM_VM_OP_pause       (18)
#define AM_VM_OP_halt        (19)
#define AM_VM_OP_fork        (20)
#define AM_VM_OP_display     (21)
#define AM_VM_OP_newline     (22)
#define AM_VM_OP_add         (23)
#define AM_VM_OP_sub         (24)
#define AM_VM_OP_mul         (25)
#define AM_VM_OP_div         (26)
#define AM_VM_OP_mod         (27)
#define AM_VM_OP_pow         (28)
#define AM_VM_OP_eq          (29)
#define AM_VM_OP_eqv         (30)
#define AM_VM_OP_equal       (31)
#define AM_VM_OP_ge          (32)
#define AM_VM_OP_le          (33)
#define AM_VM_OP_gt          (34)
#define AM_VM_OP_lt          (35)
#define AM_VM_OP_not         (36)
#define AM_VM_OP_and         (37)
#define AM_VM_OP_or          (38)
#define AM_VM_OP_isnull      (39)
#define AM_VM_OP_isundef     (40)
#define AM_VM_OP_isatom      (41)
#define AM_VM_OP_islist      (42)
#define AM_VM_OP_isnumber    (43)
#define AM_VM_OP_isnan       (44)
#define AM_VM_OP_typeof      (45)
#define AM_VM_OP_car         (46)
#define AM_VM_OP_cdr         (47)
#define AM_VM_OP_cons        (48)
#define AM_VM_OP_get_item    (49)
#define AM_VM_OP_set_item    (50)
#define AM_VM_OP_list_push   (51)
#define AM_VM_OP_list_pop    (52)
#define AM_VM_OP_length      (53)
#define AM_VM_OP_concat      (54)
#define AM_VM_OP_duplicate   (55)
#define AM_VM_OP_evalcleanup (56)
#define AM_VM_OP_dynamicwind              (57)
#define AM_VM_OP_dynamicwind_after_before (58)
#define AM_VM_OP_dynamicwind_before_after (59)
#define AM_VM_OP_dynamicwind_done         (60)
#define AM_VM_OP_wind                     (61)
// 追加到枚举末尾，勿插入中间（避免破坏模块磁盘格式中已序列化的指令编号兼容）
#define AM_VM_OP_splice                   (62) // unquote-splicing：将栈顶值打包为 UNQUOTE_SPLICING 单元素列表


// 全局内置变量
#define AM_GLOBAL_BUILTIN_VAR_NUM (37)
extern const wchar_t* AM_GLOBAL_BUILTIN_VAR[];

// 全局内置变量到 VM opcode 的映射表。
// 下标与 AM_GLOBAL_BUILTIN_VAR 一一对应；-1 表示该 builtin 没有对应 opcode。
extern const int32_t AM_BUILTIN_OPCODE_MAP[AM_GLOBAL_BUILTIN_VAR_NUM];


// 顶级词法节点、顶级作用域和顶级闭包的parent字段，用于判断上溯结束
// 注意其类型为 am_handle_t，不要与 am_value_t AM_VALUE_HANDLE_NULL 混淆
#define AM_TOP_NODE_HANDLE AM_HANDLE_NULL


// ast.var_type的值
#define AM_VAR_TYPE_OLD          (0) // 默认：普通变量（ARN换名前）
#define AM_VAR_TYPE_NEW          (1) // 普通变量（ARN换名后）
#define AM_VAR_TYPE_BUILTIN      (2) // 全局内置符号（不ARN）
#define AM_VAR_TYPE_IMPORT_REF   (3) // 导入模块符号引用（不ARN）：点号分隔的引用了外部import模块的变量，例如Mod.foo
#define AM_VAR_TYPE_NATIVE_REF   (4) // 本地模块符号引用（不ARN）：点号分隔的对native函数的调用，例如"Math.exp"
#define AM_VAR_TYPE_EXT_REF     (34) // 点号分割形式：实际上就是 AM_VAR_TYPE_IMPORT_REF 或者 AM_VAR_TYPE_NATIVE_REF，用于暂时无法确定是哪种的情况
#define AM_VAR_TYPE_IMPORT_ALIAS (5) // 导入模块的别名（不ARN）：也就是(import Mod "mod.scm")中的Mod
#define AM_VAR_TYPE_NATIVE_ID    (6) // 本地模块名（不ARN）：也就是(native Math)中的Math
#define AM_VAR_TYPE_ILTEMP       (7) // 编译过程引入的临时中间变量，AST中不存在
#define AM_VAR_TYPE_GLOBAL_FREE  (8) // 用于eval：全局无所属作用域的自由变量，普通代码属于错误，但evalee中应特殊处理


// AST数据结构
typedef struct am_ast_t {
    wchar_t *absolute_path;      // 模块代码文件所在的文件系统绝对路径
    wchar_t *module_id;          // 模块ID，从absolute_path转换而来

    wchar_t *code;               // 一切字符串的总源头
    am_token_t *tokens;          // Lexer输出的token列表
    size_t token_count;          // token数量

    am_vocab_t *symbol_vocab;    // 保存所有的symbol字符串集合，以其index为am_symbol_t
    am_vocab_t *var_vocab;       // 保存所有的变量字符串集合，以其index为am_varid_t
    am_list_t  *var_type;        // 记录每个变量的类型（取值为AM_VAR_TYPE_*），其index即为var_vocab的index

    am_allocator_t *alloc;       // 编译阶段AST专用的内存分配器
    am_heap_t *nodes;            // AST临时堆，保存编译阶段所有数据对象（包括SList也就是AST节点、词法作用域、var/sym表等）的临时堆，它们之间都是通过handle互相引用，建立起树结构
    am_map_t *node_token_mapping; // 记录AST节点把柄与token索引的映射关系（对应TS的nodeIndexes）
    am_strindex_t *strindex;     // 用于全局字符串驻留的多值哈希表，检查某个字符串（的哈希值）是否已存在于nodes

    am_map_t *scopes;            // 词法作用域：Map<handle(lambda), handle(scope)>
    am_map_t *var_arn_mapping;   // 变量ARN（Alpha-renaming）前后的映射：Map<varid, varid>，key是ARN后的新varid，value是ARN前的旧varid

    am_handle_t top_lambda_handle; // 最顶层lambda节点的把柄（通过am_ast_get_top_lambda_node_handle计算）
    am_list_t *lambda_handles;   // 记录所有的lambda节点的把柄（对应TS的lambdaHandles）
    am_list_t *tailcall_handles; // 记录所有的尾调用节点的把柄（对应TS的tailcall）
    am_list_t *var_top;          // 顶级变量varid列表（即顶层作用域define的变量）（对应TS的topVariables）
    am_map_t *dependencies;      // 依赖模块记录：Map<varid, handle>（对应TS的dependencies）根据(import mod_alias "path/to/mod.scm")记录
    am_map_t *natives;           // 本地库记录：Map<varid, handle>（对应TS的natives）根据(native Math)记录，其中handle可暂时设置为AM_VALUE_HANDLE_NULL备用

    size_t opstack_depth;        // 静态分析得到的最大opstack栈深度（在link后最后分析）
} am_ast_t;


// 功能描述：创建AST对象。调用者保留code、absolute_path、tokens的所有权，AST只保存指针。
// 实现说明：成功返回AST指针，失败返回NULL。
am_ast_t *am_ast_create(am_allocator_t *alloc, wchar_t *code, wchar_t *absolute_path, am_token_t *tokens, size_t token_count);


// 功能描述：销毁AST对象，释放AST自身及其内部所有堆对象。
// 实现说明：成功返回0，失败返回-1。注意不释放调用者传入的code、absolute_path、tokens。
int32_t am_ast_destroy(am_ast_t *ast);


// 功能描述：深拷贝AST对象（对应TS的AST.Copy）
// 实现说明：创建新的AST，深拷贝所有内部集合和堆对象。code、absolute_path、tokens与源AST共享指针。
am_ast_t *am_ast_copy(am_ast_t *ast);


// 功能描述：设置AST节点把柄对应的token索引。
// 实现说明：成功返回0，失败返回-1。
int32_t am_ast_set_node_token_index(am_ast_t *ast, am_handle_t node_handle, size_t token_index);


// 功能描述：获取AST节点把柄对应的token索引（对应TS的nodeIndexes.get）。
// 实现说明：若不存在，返回SIZE_MAX。
size_t am_ast_get_node_token_index(am_ast_t *ast, am_handle_t node_handle);


// 功能描述：将importee融合进importer，也就是importer吃掉importee。
// 实现说明：成功返回0；失败返回-1。
int32_t am_ast_merge(am_ast_t *importer, am_ast_t *importee, int32_t order);


// 功能描述：遍历tokens，使用其中的KEYWORD和SYMBOL构建ast->symbol_vocab，同时等于是注册了am_symbol_t，并将am_symbol_t记录在token中
// 实现说明：返回symbol总数。注意将object.h中定义的24个Keyword置于词典的前24个条目。
size_t am_build_symbol_vocabulary(am_ast_t *ast);


// 功能描述：遍历tokens，使用其中的VARIABLE构建ast->var_vocab，同时等于是注册了am_varid_t，并将varid记录在token中。
// 实现说明：返回varid总数。
size_t am_build_variable_vocabulary(am_ast_t *ast);




// 功能描述：判断某个变量在形式上是否是“前缀.后缀”的格式（统称为EXT_REF，外部引用格式），是返回0，否返回-1
// 设计说明：parse和ARN阶段，这种形式的变量可能是AM_VAR_TYPE_IMPORT_REF或AM_VAR_TYPE_NATIVE_REF，保留原形，不参与ARN。
// 实现说明：(varid)--[ast->var_vocab]-->var_str-->判断其是否是被唯一点号分成两部分的形式（只有一个“.”，且不在开头和末尾）
int32_t am_ast_check_ext_ref(am_ast_t *ast, am_varid_t v);


// 功能描述：判断某个变量是否是AM_VAR_TYPE_NATIVE_REF，也就是对本地宿主库native的调用，是返回0，否返回-1（对应TS的AST.IsNativeCall）
// 实现说明：(varid)(t.id)--[ast->var_vocab]-->var_str-->提取点号分隔的第1部分--[ast->var_vocab]-->native_varid--[ast->natives]-->是否存在
int32_t am_ast_check_native_ref(am_ast_t *ast, am_varid_t v);


// 功能描述：判断某个变量是否是AM_VAR_TYPE_IMPORT_REF，即导入模块的外部引用（“别名.标识符”的格式），是返回0，否返回-1
// 设计说明：外部引用：指的是通过import和点号分隔标识符，引用外部模块变量。(import Alias "/path/to/module.scm")表达式，声明对外部模块的导入，并赋予其“别名”Alias，别名属于特殊变量，其类型为AM_VAR_TYPE_IMPORT_ALIAS。代码中通过“别名.标识符”的格式，引用外部模块的变量。“别名.标识符”整体也是一个变量，在parse阶段，其类型为AM_VAR_TYPE_IMPORT_REF。
// 实现说明：(varid)--[ast->var_vocab]-->var_str-->提取最后一个点号分隔的第1部分--[ast->var_vocab]-->alias_varid--[ast->dependencies]-->是否存在
int32_t am_ast_check_import_ref(am_ast_t *ast, am_varid_t v);


// 功能描述：根据把柄，从AST->nodes堆中获取相应的am_value_t（由调用者解包并使用）（对应TS的AST.GetNode）
// 设计说明：解释器的值-对象映射机制比较复杂。就本函数来说，根据handle从nodes堆中获得了am_value_t，这是一个打包后的值（因为只有打包后才能装进map、list等容器）。调用者通过该函数获得了am_value_t之后，应当自行判断其类型并解包。例如，如果调用者期望通过handle从nodes堆中获得一个am_object_t的指针，则通过本函数获得am_value_t后，将其作为am_object_t*也就是AM_VALUE_TYPE_PTR进行解包（使用am_value_to_ptr），这样就可以通过解包得到的ptr直接在C语言层面访问allocator管理的内存中（也就是被AST->nodes堆所封装起来的内存）的object对象。
am_value_t am_ast_get_node(am_ast_t *ast, am_handle_t handle);


// 功能描述：创建lambda对象，返回其在AST->nodes堆中的把柄（对应TS的AST.MakeLambdaNode）
// 实现说明：先从heap中申请一个把柄，再创建一个类型为AM_LIST_TYPE_LAMBDA的am_obj_list_t对象，以32为初始容量，再将对象指针打包成am_value_t与已分配把柄绑定在一起。同时，在ast->lambda_handles中登记这个把柄。最后返回把柄。如有异常情况，返回空把柄AM_HANDLE_NULL，以示失败。
am_handle_t am_ast_make_lambda_node(am_ast_t *ast, am_handle_t parent);


// 功能描述：创建SList对象，返回其在AST->nodes堆中的把柄（对应TS的AST.MakeApplicationNode）
// 实现说明：先从heap中申请一个把柄，再创建一个类型为type=AM_LIST_TYPE_APPLICATION/AM_LIST_TYPE_QUOTE/AM_LIST_TYPE_QUASIQUOTE/AM_LIST_TYPE_UNQUOTE/AM_LIST_TYPE_UNQUOTE_SPLICING的am_obj_list_t对象，以32为初始容量，再将对象指针打包成am_value_t与已分配把柄绑定在一起。最后返回把柄。如有异常情况，返回空把柄AM_HANDLE_NULL，以示失败。
am_handle_t am_ast_make_slist_node(am_ast_t *ast, am_handle_t parent, int32_t type);


// 功能描述：创建WString对象，返回其在AST->nodes堆中的把柄（对应TS的AST.MakeStringNode）
// 实现说明：先从AST->nodes中申请一个把柄，再根据AM_TOKEN_TYPE_STRING类型的am_token_t t 所表示的字符串（注意：可以根据其指示的index和length从ast->code中获取），创建一个am_obj_wstring_t对象，再将对象指针打包成am_value_t与已分配把柄绑定在一起。最后返回把柄。如有异常情况，返回空把柄AM_HANDLE_NULL，以示失败。
am_handle_t am_ast_make_wstring_node(am_ast_t *ast, am_token_t *str_token);


// 功能描述：查找AST->nodes堆中最顶级am_obj_list_t对象的handle，也就是parent字段为AM_HANDLE_NULL的am_obj_list_t对象。（对应TS的AST.TopApplicationNodeHandle）
// 设计说明：根据编译器的约定，合法Scheme代码的顶层结构应当是一个thunk的调用，即((lambda () ...))，这个函数就是用来获取这个顶层APPLICATION的。
// 实现说明：如有异常情况，返回空把柄AM_HANDLE_NULL，以示失败。
am_handle_t am_ast_get_top_node_handle(am_ast_t *ast);


// 功能描述：查找AST->nodes堆中顶级am_obj_list_t（Lambda）对象的handle，也就是最顶级application list对象的第一个child。（对应TS的AST.TopLambdaNodeHandle）
// 设计说明：根据编译器的约定，合法Scheme代码的顶层结构应当是一个thunk的调用，即((lambda () ...))，这个函数就是用来获取这个顶层APPLICATION的第一个child也就是顶层lambda（thunk）的。
// 实现说明：如有异常情况，返回空把柄AM_HANDLE_NULL，以示失败。
am_handle_t am_ast_get_top_lambda_node_handle(am_ast_t *ast);


// 功能描述：获取位于全局作用域的node列表（也就是函数体列表）。（对应TS的AST.GetGlobalNodes）
// 设计说明：取am_ast_get_top_lambda_node_handle也就是顶层lambda（thunk）的bodies，返回一个am_value_t的数组，由调用者负责解包、解释、释放。
// 实现说明：如有异常情况，返回NULL，以示失败。
am_value_t *am_ast_get_global_nodes(am_ast_t *ast);



// 功能描述：设置全局作用域（顶层lambda）的node列表（也就是函数体列表）。（对应TS的AST.SetGlobalNodes）
// 设计说明：用bodies整体替换am_ast_get_top_lambda_node_handle也就是顶层lambda（thunk）的bodies。
// 实现说明：通过am_list_lambda_set_bodies实现，这个过程可能涉及lambda对象指针的变化，如有变化，则更新AST->nodes中对应handle的值（打包成am_value_t的）。n_body为bodies数组的长度。如有扩容失败等异常情况，返回-1。执行成功则返回0。
int32_t am_ast_set_global_nodes(am_ast_t *ast, am_value_t *bodies, size_t n_body);




// 功能描述：从某个节点开始，向上上溯查找某个varid归属的lambda节点把柄，也就是该varid作为哪个lambda节点的parameter（对应TS的Analyser.ts中的searchVarLambdaHandle）
// 设计说明：该函数用于“变量换名”过程，旨在寻找其最近上级lambda节点的把柄，进而确定其所在的词法作用域。该函数依赖于AST第一趟扫描“作用域分析”的结果。
// 实现说明：该函数的输入是变量换名前的varid，以及上溯查找起点节点的handle。如有异常情况，返回空把柄AM_HANDLE_NULL，以示失败。
am_handle_t am_ast_find_var_lambda_handle(am_ast_t *ast, am_varid_t varid, am_handle_t from_node_handle);


// 功能描述：从某个节点开始，向上上溯查找最近的lambda节点的把柄（对应TS的Analyser.ts中的nearestLambdaHandle）
// 设计说明：该函数用于确定某个节点最近上级lambda节点的把柄，进而确定其所在的词法作用域。
// 实现说明：该函数的输入是上溯查找起点节点的handle。如有异常情况，返回空把柄AM_HANDLE_NULL，以示失败。
am_handle_t am_ast_find_nearest_lambda_handle(am_ast_t *ast, am_handle_t from_node_handle);


// 功能表述：生成模块（AST）内唯一的变量名（对应TS的Analyser.ts中的MakeUniqueVariable）
// 设计说明：该函数用于“变量换名”阶段，用于生成携带作用域信息的、全局唯一的变量名，并将其新增注册到ast->var_vocab中。
// 实现说明：基于varid和所在lambda节点的handle，生成一个新的变量名字符串。规则是："V.module_id.lambda_handle.var_string"，并将其新增注册到ast->var_vocab，返回值是新变量名的ast->var_vocab的index。如有异常情况，返回SIZE_MAX，以示失败。
am_varid_t am_ast_make_unique_variable(am_ast_t *ast, am_varid_t varid, am_handle_t lambda_handle);


// 功能描述：为 import 别名生成模块级唯一变量名（module_id.alias），并将其注册到 ast->var_vocab，同时设置其 var_type 为 AM_VAR_TYPE_IMPORT_ALIAS。
// 实现说明：成功返回新的 varid，失败返回 SIZE_MAX。
am_varid_t am_ast_make_unique_module_alias(am_ast_t *ast, am_varid_t alias_varid);


// 功能描述：为 import 外部引用生成模块级唯一变量名（module_id.import_ref），并将其注册到 ast->var_vocab，同时设置其 var_type 为 AM_VAR_TYPE_IMPORT_REF。
// 实现说明：成功返回新的 varid，失败返回 SIZE_MAX。
am_varid_t am_ast_make_unique_import_ref(am_ast_t *ast, am_varid_t import_ref_varid);


// 功能描述：向 tailcall_handles 中添加一个尾调用节点把柄。
// 实现说明：成功返回0，失败返回-1。
int32_t am_ast_add_tailcall(am_ast_t *ast, am_handle_t handle);


// 功能描述：向 var_top 中添加一个顶级变量 varid。
// 实现说明：成功返回0，失败返回-1。
int32_t am_ast_add_var_top(am_ast_t *ast, am_varid_t varid);


// 功能描述：设置依赖模块记录。
// 实现说明：alias_varid 为 import 语句中模块别名对应的 varid；path_handle 为模块路径字符串节点在 ast->nodes 中的把柄。成功返回0，失败返回-1。
int32_t am_ast_set_dependency(am_ast_t *ast, am_varid_t alias_varid, am_handle_t path_handle);


// 功能描述：设置本地库记录。
// 实现说明：native_varid 为 native 语句中库名对应的 varid；handle 可暂时设置为 AM_VALUE_HANDLE_NULL。成功返回0，失败返回-1。
int32_t am_ast_set_native(am_ast_t *ast, am_varid_t native_varid, am_handle_t handle);


// 功能描述：为lambda节点设置对应的词法作用域把柄。
// 实现说明：成功返回0，失败返回-1。
int32_t am_ast_set_scope(am_ast_t *ast, am_handle_t lambda_handle, am_handle_t scope_handle);


// 功能描述：获取lambda节点对应的词法作用域把柄。
// 实现说明：若不存在，返回 AM_HANDLE_NULL。
am_handle_t am_ast_get_scope(am_ast_t *ast, am_handle_t lambda_handle);








// 功能描述：将模块绝对路径转换为模块ID。
// 实现说明：规则见 AGENTS.md。返回使用 ast 分配器分配的 wchar_t*，失败返回 NULL。
wchar_t *am_absolute_path_to_module_id(am_allocator_t *alloc, const wchar_t *absolute_path);


// 功能描述：将AST中的某个节点转成Scheme代码字符串（对应TS的AST.NodeToString）。
// 实现说明：返回使用 alloc 分配器分配的以 L'\0' 结尾的宽字符串，失败返回 NULL。
//         若 length 不为 NULL，则将字符串的逻辑长度（字符数）写入 *length。
wchar_t *am_ast_node_to_string(am_allocator_t *alloc, am_ast_t *ast, am_handle_t node_handle, size_t *length);


#ifdef __cplusplus
}
#endif

#endif
