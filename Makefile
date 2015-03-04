OBJS := hand.o
TARGET := hand
CFLAGS := -Wall
LDFLAGS := -lsupc++ -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_video -lm

all: $(OBJS)
	$(CC) -o $(TARGET) $(OBJS) $(LDFLAGS)

clean:
	rm -f $(OBJS) $(TARGET) video.avi

