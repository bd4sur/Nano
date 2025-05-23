#ifndef __NANO_INFER_MATMUL_PT__
#define __NANO_INFER_MATMUL_PT__

#include <pthread.h>
#include <semaphore.h>
#include <unistd.h>
#include <stdlib.h>

#define MIN_CHUNK_SIZE (102)

// 线程池结构体
typedef struct {
    pthread_t* threads;
    int num_threads;
    struct Task* task_head;
    struct Task* task_tail;
    pthread_mutex_t mutex;
    pthread_cond_t cond;
    int shutdown;
} ThreadPool;

// 任务结构体
typedef struct Task {
    void (*func)(void*);
    void* arg;
    struct Task* next;
} Task;

// 线程参数结构体
typedef struct {
    float* xout;
    float* x;
    float* w;
    int n;
    int start_i;
    int end_i;
    sem_t* sem; // 完成信号量
} ThreadArgs;


void matmul_pthread(float* xout, float* x, float* w, int n, int d);
void matmul_pthread_cleanup();

#endif
