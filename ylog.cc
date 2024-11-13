#include "ylog.h"

char time_str[80];
char * ylog_get_time()
{
    time_t rawtime;
    struct tm *timeinfo;

    time(&rawtime);
    timeinfo = localtime(&rawtime);

    strftime(time_str, sizeof(time_str), "%Y-%m-%d %H:%M:%S", timeinfo);
    return time_str;
}

int ylog_now_print()
{
	printf("%s", ylog_get_time());
	return 0;
}


#if YLOG_FILE
int g_fd;
int ylog_file_init()
{
	char tmp_file[94];
	snprintf(tmp_file, 94, "%s", ylog_get_time());
	g_fd = open(tmp_file, O_CREAT | O_RDWR, 0666);
	if (g_fd == -1) {
		perror("error open file\n");
		return 1;
	} else {
		printf("create %s log success\n", tmp_file);
	}
	space_check_pthread();
	return 0;
}

int ylog_file_close()
{
	close(g_fd);
	return 0;
}
#endif

int ylog_init()
{
#if YLOG_FILE
	ylog_file_init();
#endif
	return 0;
}

int ylog_deinit()
{
#if YLOG_FILE
	ylog_file_close();
#endif
	return 0;
}

YLOG_CB ylog_cb;
void register_cb(YLOG_CB user_cb)
{
	ylog_cb = user_cb;
}
void *monitorDiskSpace(void *arg)
{
	while (1) {
		struct statvfs fs_info;
		if (statvfs("/", &fs_info) == 0) {
			long long availableBlocks = fs_info.f_bavail;
			long long blockSize = fs_info.f_frsize;
			long long availableSize = availableBlocks * blockSize;

			if (availableSize < MAX_SPACE) {
				ylog("warning: space < %d!\n", MAX_SPACE);
				if (ylog_cb != NULL)
					ylog_cb(NULL);
			}
		} else {
			ylog("get space error\n");
		}
		sleep(5);
	}
	return NULL;
}

int space_check_pthread()
{
	pthread_t thread_id;
	if (pthread_create(&thread_id, NULL, monitorDiskSpace, NULL)!= 0) {
		ylog("create phtread error\n");
		return 1;
	}

	//pthread_join(thread_id, NULL);

	return 0;
}
