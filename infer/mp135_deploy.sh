#!/bin/bash
# =============================================================================
# CoreMP135（STM32MP135）构建 + 部署脚本
#
#   ./mp135_deploy.sh            # 构建 + 部署
#   ./mp135_deploy.sh build      # 只交叉编译，产物 = .build-mp135/nano_pod
#   ./mp135_deploy.sh deploy     # 只部署已有产物
#
# 目标机上的落地物（全部由本脚本生成，不依赖仓库里的额外文件）：
#   $RBIN                        可执行文件（默认 /root/nano_pod）
#   /etc/init.d/S99nanopod       开机自启脚本（内容内联在下方 heredoc 里）
#   /etc/rc.local                自动注释掉旧的手工启动行（幂等）
#   $RDIR/app, dict, ebook, ...  App 源码与业务资源
#
# 环境变量（均有默认值）：
#   RHOST=192.168.10.39      板子 IP——DHCP 每次重启都会换，部署前请确认
#   RUSER=root
#   RDIR=/home/bd4sur/linglong  业务资源根目录（= platform.h 的 PLATFORM_ROOT_DIR）
#   RBIN=/root/nano_pod         可执行文件落点（与 S99nanopod 的 DAEMON 一致）
#   RES_SRC=/home/bd4sur/linglong  业务资源母本（本机，非仓库内容；缺哪目录就跳过）
#   SSH_OPTS                    额外 ssh 参数，默认 -o StrictHostKeyChecking=no
#
# 工具链：mp135/ 下的 Linaro GCC 7.5（原 mp135.mk 指向的 buildroot output/host
# 里没有 gcc / sysroot，故本脚本改用独立工具链 + 源码自带 tinyalsa 静态库）。
# =============================================================================
set -euo pipefail

INFER=$(cd "$(dirname "$0")" && pwd)

TCROOT=${TCROOT:-/home/bd4sur/mp135/tc/gcc-linaro-7.5.0-2019.12-x86_64_arm-linux-gnueabihf}
CROSS=$TCROOT/bin/arm-linux-gnueabihf-
CC=${CROSS}gcc
AR=${CROSS}ar

BUILD=$INFER/.build-mp135
OBJ=$BUILD/obj
TAL=$BUILD/tinyalsa
BIN=$BUILD/nano_pod

RHOST=${RHOST:-192.168.10.39}
RUSER=${RUSER:-root}
RDIR=${RDIR:-/home/bd4sur/linglong}
RBIN=${RBIN:-/root/nano_pod}
RES_SRC=${RES_SRC:-/home/bd4sur/linglong}
SSH_OPTS=${SSH_OPTS:--o StrictHostKeyChecking=no}

ARCH_FLAGS="-mcpu=cortex-a7 -mfpu=neon-vfpv4 -mfloat-abi=hard -mthumb"
DEFS="-DNANO_POD_MP135 -DAM_NATIVE_MAX_LIBS=32"
INC="-I$INFER -I$INFER/vendor/Animac -I$INFER/toolchains/tinyalsa/include"
OPT="-O3 -ffast-math -Wall"
JOBS=$(nproc)

# 部署的目录清单：infer/app/ → $RDIR/app/，$RES_SRC/<d>/ → $RDIR/<d>/
# 注：app/ 与资源目录都用 rsync --delete 镜像；data/ 下的 ilcache/ 是运行时索引缓存，
#     被清掉后首次进词典会重建，属预期行为。
APP_SRC=$INFER/app
RES_DIRS="data dict ebook icon llm music wallpaper"

ssh_do() { ssh $SSH_OPTS "$RUSER@$RHOST" "$@"; }
rsync_do() { rsync -a --delete -e "ssh $SSH_OPTS" "$@"; }

# -----------------------------------------------------------------------------
# 构建
# -----------------------------------------------------------------------------
build() {
    [ -x "$CC" ] || { echo "找不到交叉编译器：$CC" >&2; exit 1; }
    mkdir -p "$OBJ" "$TAL"

    # 1) tinyalsa 静态库（源码内自带，直接编进二进制，目标机无需装 libtinyalsa）
    if [ ! -f "$TAL/libtinyalsa.a" ]; then
        echo "== 编译 tinyalsa =="
        for f in "$INFER"/toolchains/tinyalsa/src/*.c; do
            $CC $ARCH_FLAGS -O2 -fPIC -I"$INFER/toolchains/tinyalsa/include" \
                -c "$f" -o "$TAL/$(basename "${f%.c}").o"
        done
        $AR rcs "$TAL/libtinyalsa.a" "$TAL"/*.o
    fi

    # 2) 源文件清单取自 mp135.mk 的 pod 目标，避免两份清单漂移
    SRCS=$(cd "$INFER" && make -f mp135.mk -n pod 2>/dev/null \
            | grep -- '-DNANO_POD_MP135' | head -1 \
            | tr ' ' '\n' | grep '\.c$' | sed "s|^|$INFER/|")
    echo "== 编译 nano_pod（$(echo "$SRCS" | wc -l) 个源文件，-j$JOBS）=="

    compile_one() {
        src=$1
        rel=${src#$INFER/}
        obj=$OBJ/$(echo "$rel" | tr '/' '_' | sed 's/\.c$/.o/')
        [ -f "$obj" ] && [ "$obj" -nt "$src" ] && [ -z "${FORCE:-}" ] && return 0
        $CC $ARCH_FLAGS $DEFS $OPT $INC -c "$src" -o "$obj" || { echo "FAIL: $rel" >&2; return 1; }
    }
    export -f compile_one
    export CC OBJ INFER ARCH_FLAGS DEFS OPT INC
    echo "$SRCS" | xargs -P "$JOBS" -I{} bash -c 'compile_one "$@"' _ {}

    # 3) 链接 + strip
    OBJS=$(echo "$SRCS" | sed "s|$INFER/||" | tr '/' '_' | sed 's/\.c$/.o/' | sed "s|^|$OBJ/|")
    echo "== 链接 =="
    $CC $ARCH_FLAGS $OBJS -o "$BIN" -lm -lpthread -ldl "$TAL/libtinyalsa.a" -Wl,--gc-sections
    ${CROSS}strip --strip-unneeded "$BIN"
    file "$BIN"
    ls -lh "$BIN"
}

# -----------------------------------------------------------------------------
# 部署
# -----------------------------------------------------------------------------
deploy() {
    [ -f "$BIN" ] || { echo "没有构建产物 $BIN（先跑 $0 build）" >&2; exit 1; }
    echo "== 部署到 $RUSER@$RHOST =="

    # 1) 可执行文件（留一份上一版备查）
    ssh_do "cp -a '$RBIN' '$RBIN.old' 2>/dev/null; mkdir -p '$RDIR/app'; true"
    rsync_do "$BIN" "$RUSER@$RHOST:$RBIN"

    # 2) App 源码（镜像同步，会清掉已删除的 .scm）
    rsync_do "$APP_SRC/" "$RUSER@$RHOST:$RDIR/app/"

    # 3) 业务资源目录（母本 $RES_SRC 在仓库外，缺哪目录就跳过）
    for d in $RES_DIRS; do
        if [ -d "$RES_SRC/$d" ]; then
            printf '   %-10s ' "$d"
            rsync_do "$RES_SRC/$d/" "$RUSER@$RHOST:$RDIR/$d/"
            ssh_do "du -sh '$RDIR/$d' | cut -f1"
        else
            echo "   $d: 本机无 $RES_SRC/$d，跳过"
        fi
    done

    # 4) 开机自启脚本（内联生成，不单独跟踪文件）
    echo "== 安装 /etc/init.d/S99nanopod（DAEMON=$RBIN）=="
    local init_tmp
    init_tmp=$(mktemp)
    cat >"$init_tmp" <<'INIT_EOF'
#!/bin/sh
#
# Nano-Pod（CoreMP135）开机自启
# 由本仓库 infer/mp135_deploy.sh 生成，请勿手工修改（改动请改脚本后重跑）。
# 由 /etc/init.d/rcS 在 S90rclocal 之后启动，此时网络已就绪。
# 手动控制：/etc/init.d/S99nanopod {start|stop|restart|status}

DAEMON=__DAEMON__
PIDFILE=/var/run/nano_pod.pid
LOGFILE=/var/log/nano_pod.log

start() {
	printf "Starting nano_pod: "
	if [ ! -x "$DAEMON" ]; then
		echo "FAIL (not executable: $DAEMON)"
		return 1
	fi
	if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
		echo "already running (pid $(cat "$PIDFILE"))"
		return 0
	fi
	cd /root || cd /
	setsid "$DAEMON" >>"$LOGFILE" 2>&1 &
	echo $! > "$PIDFILE"
	sleep 2
	if kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
		echo "OK (pid $(cat "$PIDFILE"))"
		return 0
	fi
	echo "FAIL (see $LOGFILE)"
	return 1
}

stop() {
	printf "Stopping nano_pod: "
	PIDS=$(pidof nano_pod)
	if [ -z "$PIDS" ]; then
		echo "not running"
		rm -f "$PIDFILE"
		return 0
	fi
	kill $PIDS 2>/dev/null
	i=0
	while [ $i -lt 10 ] && [ -n "$(pidof nano_pod)" ]; do
		sleep 1
		i=$((i + 1))
	done
	PIDS=$(pidof nano_pod)
	[ -n "$PIDS" ] && kill -9 $PIDS 2>/dev/null
	rm -f "$PIDFILE"
	echo "OK"
	return 0
}

case "$1" in
start)
	start
	;;
stop)
	stop
	;;
restart)
	stop
	start
	;;
status)
	PIDS=$(pidof nano_pod)
	if [ -n "$PIDS" ]; then echo "nano_pod running: $PIDS"; else echo "nano_pod stopped"; fi
	;;
*)
	echo "Usage: $0 {start|stop|restart|status}"
	exit 1
	;;
esac
INIT_EOF
    sed -i "s|__DAEMON__|$RBIN|" "$init_tmp"
    ssh_do "cat > /etc/init.d/S99nanopod && chmod +x /etc/init.d/S99nanopod" <"$init_tmp"
    rm -f "$init_tmp"

    # 5) rc.local 里旧的手工启动行改为注释（幂等；原文件首次部署时备份）
    ssh_do "cp -a /etc/rc.local /root/rc.local.bak 2>/dev/null; \
            sed -i 's|^/root/nano_pod &|# nano_pod 改由 /etc/init.d/S99nanopod 开机自启|' /etc/rc.local; \
            grep -n 'nano_pod' /etc/rc.local || true"

    # 6) 立即生效并检查
    echo "== 重启服务 =="
    ssh_do "/etc/init.d/S99nanopod restart; sleep 3; /etc/init.d/S99nanopod status; \
            md5sum '$RBIN'; df -h / | tail -1"
}

case "${1:-all}" in
build)  build ;;
deploy) deploy ;;
all)    build; deploy ;;
*)      echo "用法: $0 [build|deploy]" >&2; exit 1 ;;
esac
