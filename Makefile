OBJS := game.o
CC := g++ 
TARGET := game
CXXFLAGS := -std=c++11
LDFLAGS := -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_video -lm -lopencv_gpu

all: $(OBJS)
	$(CC) -o $(TARGET) $(CFLAGS) $(OBJS) $(LDFLAGS)

clean:
	rm -f $(OBJS) $(TARGET) video.avi

