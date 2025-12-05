#include "matmul_pthread.h"

// 全局线程池（实际使用时建议通过参数传递）
static ThreadPool* g_threadpool = NULL;

// 工作线程函数
void* worker_thread(void* arg) {
    ThreadPool* pool = (ThreadPool*)arg;
    while (1) {
        pthread_mutex_lock(&pool->mutex);
        
        // 等待任务或关闭信号
        while (!pool->shutdown && !pool->task_head) {
            pthread_cond_wait(&pool->cond, &pool->mutex);
        }
        
        if (pool->shutdown) {
            pthread_mutex_unlock(&pool->mutex);
            break;
        }
        
        // 取出任务
        Task* task = pool->task_head;
        if (task) {
            pool->task_head = task->next;
            if (!pool->task_head) pool->task_tail = NULL;
        }
        pthread_mutex_unlock(&pool->mutex);
        
        if (task) {
            task->func(task->arg);
            free(task);
        }
    }
    return NULL;
}

// 初始化线程池
ThreadPool* threadpool_init(int num_threads) {
    ThreadPool* pool = (ThreadPool*)malloc(sizeof(ThreadPool));
    pool->num_threads = num_threads;
    pool->task_head = pool->task_tail = NULL;
    pool->shutdown = 0;
    pthread_mutex_init(&pool->mutex, NULL);
    pthread_cond_init(&pool->cond, NULL);
    
    pool->threads = (pthread_t*)malloc(num_threads * sizeof(pthread_t));
    for (int i = 0; i < num_threads; i++) {
        pthread_create(&pool->threads[i], NULL, worker_thread, pool);
    }
    return pool;
}

// 提交任务到线程池
void threadpool_submit(ThreadPool* pool, void (*func)(void*), void* arg) {
    Task* task = (Task*)malloc(sizeof(Task));
    task->func = func;
    task->arg = arg;
    task->next = NULL;
    
    pthread_mutex_lock(&pool->mutex);
    if (pool->task_tail) {
        pool->task_tail->next = task;
    } else {
        pool->task_head = task;
    }
    pool->task_tail = task;
    pthread_cond_signal(&pool->cond);
    pthread_mutex_unlock(&pool->mutex);
}

// 销毁线程池
void threadpool_destroy(ThreadPool* pool) {
    pthread_mutex_lock(&pool->mutex);
    pool->shutdown = 1;
    pthread_cond_broadcast(&pool->cond);
    pthread_mutex_unlock(&pool->mutex);
    
    for (int i = 0; i < pool->num_threads; i++) {
        pthread_join(pool->threads[i], NULL);
    }
    
    free(pool->threads);
    pthread_mutex_destroy(&pool->mutex);
    pthread_cond_destroy(&pool->cond);
    free(pool);
}

// 任务处理函数
void matmul_task(void* arg) {
    ThreadArgs* args = (ThreadArgs*)arg;
    for (int i = args->start_i; i < args->end_i; i++) {
        float val = 0.0f;
        for (int j = 0; j < args->n; j++) {
            val += args->w[i * args->n + j] * args->x[j];
        }
        args->xout[i] = val;
    }
    sem_post(args->sem); // 标记任务完成
    free(args);
}

void matmul_pthread(float* xout, float* x, float* w, int n, int d) {
    // 初始化全局线程池（首次调用时）
    if (!g_threadpool) {
        int num_cores = sysconf(_SC_NPROCESSORS_ONLN);
        g_threadpool = threadpool_init(num_cores > 0 ? num_cores : 4);
    }
    
    // 动态确定任务粒度（每个任务至少处理MIN_CHUNK_SIZE行）
    const int min_chunk = MIN_CHUNK_SIZE;
    int num_tasks = (d + min_chunk - 1) / min_chunk;
    num_tasks = num_tasks > 0 ? num_tasks : 1;
    
    sem_t sem;
    sem_init(&sem, 0, 0);
    
    int current_start = 0;
    for (int t = 0; t < num_tasks; t++) {
        int chunk_size = min_chunk;
        if (t == num_tasks - 1) {
            chunk_size = d - current_start;
        }
        
        ThreadArgs* args = (ThreadArgs*)malloc(sizeof(ThreadArgs));
        args->xout = xout;
        args->x = x;
        args->w = w;
        args->n = n;
        args->start_i = current_start;
        args->end_i = current_start + chunk_size;
        args->sem = &sem;
        
        current_start += chunk_size;
        threadpool_submit(g_threadpool, matmul_task, args);
    }
    
    // 等待所有任务完成
    for (int t = 0; t < num_tasks; t++) {
        sem_wait(&sem);
    }
    sem_destroy(&sem);
}

// 任务处理函数
void matmul_quant_task(void* arg) {
    ThreadArgsQuant* args = (ThreadArgsQuant*)arg;
    for (int i = args->start_i; i < args->end_i; i++) {
        float val = 0.0f;
        int32_t ival = 0;
        int in = i * args->n;
        for (int j = 0; j <= args->n - GS; j += GS) {
            for (int k = 0; k < GS; k++) {
                ival += ((int32_t) (args->x)->q[j + k]) * ((int32_t) (args->w)->q[in + j + k]);
            }
            val += ((float) ival) * (args->w)->s[(in + j) / GS] * (args->x)->s[j / GS];
            ival = 0;
        }

        args->xout[i] = val;
    }
    sem_post(args->sem); // 标记任务完成
    free(args);
}

void matmul_quant_pthread(float* xout, Q80_Tensor *x, Q80_Tensor *w, int n, int d) {
    // 初始化全局线程池（首次调用时）
    if (!g_threadpool) {
        int num_cores = sysconf(_SC_NPROCESSORS_ONLN);
        g_threadpool = threadpool_init(num_cores > 0 ? num_cores : 4);
    }
    
    // 动态确定任务粒度（每个任务至少处理MIN_CHUNK_SIZE行）
    const int min_chunk = MIN_CHUNK_SIZE;
    int num_tasks = (d + min_chunk - 1) / min_chunk;
    num_tasks = num_tasks > 0 ? num_tasks : 1;
    
    sem_t sem;
    sem_init(&sem, 0, 0);
    
    int current_start = 0;
    for (int t = 0; t < num_tasks; t++) {
        int chunk_size = min_chunk;
        if (t == num_tasks - 1) {
            chunk_size = d - current_start;
        }
        
        ThreadArgsQuant* args = (ThreadArgsQuant*)malloc(sizeof(ThreadArgsQuant));
        args->xout = xout;
        args->x = x;
        args->w = w;
        args->n = n;
        args->start_i = current_start;
        args->end_i = current_start + chunk_size;
        args->sem = &sem;
        
        current_start += chunk_size;
        threadpool_submit(g_threadpool, matmul_quant_task, args);
    }
    
    // 等待所有任务完成
    for (int t = 0; t < num_tasks; t++) {
        sem_wait(&sem);
    }
    sem_destroy(&sem);
}

/* 程序结束时调用 */
void matmul_pthread_cleanup() {
    if (g_threadpool) {
        threadpool_destroy(g_threadpool);
        g_threadpool = NULL;
    }
}
