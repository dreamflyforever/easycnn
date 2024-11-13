#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <fcntl.h>
#include <pthread.h>
#include <unistd.h>
#include <sys/statvfs.h>

/*save log*/
#define YLOG_FILE 1
/*consol ouput*/
#define DEBUG 0

#define MAX_SPACE  50 * 1024 * 1024

typedef void (*YLOG_CB) (void *p_arg);

extern int ylog_init();
extern int ylog_deinit();
extern char * ylog_get_time();
extern int ylog_now_print();

#define DEBUG 0
#if DEBUG
#define ylog(format, ...) \
	{ylog_now_print(); printf(" [%s : %s : %d] ", \
	__FILE__, __func__, __LINE__); \
	printf(format, ##__VA_ARGS__);}
#elif YLOG_FILE

extern int g_fd;
#define ylog(format, ...) \
{ \
	char tmp[512] = {0}; \
	snprintf(tmp, 128, "%s [%s : %s : %d] ", ylog_get_time(),  __FILE__, __func__, __LINE__); \
	write(g_fd, tmp, strlen(tmp));\
	memset(tmp, 0, 512); \
	snprintf(tmp, 512, format, ##__VA_ARGS__); \
	write(g_fd, tmp, strlen(tmp)); }

#else
#define ylog(format, ...) 
#endif

extern int space_check_pthread();

void register_cb(YLOG_CB user_cb);
