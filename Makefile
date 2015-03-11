OBJS := hand.o
CC := g++
TARGET := hand
CFLAGS := -Wall
LDFLAGS := -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_video -lm -lopencv_gpu

all: $(OBJS)
	$(CC) -o $(TARGET) $(OBJS) $(LDFLAGS)

clean:
	rm -f $(OBJS) $(TARGET) video.avi

