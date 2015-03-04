/*
 * Simple hand detection algorithm based on OpenCV
 *
 * (C) Copyright 2012-2013 <b.galvani@gmail.com>
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License as
 * published by the Free Software Foundation; either version 2 of
 * the License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 */

#include <stdio.h>

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/gpu/gpu.hpp"

#define VIDEO_FILE	"video.avi"
#define VIDEO_FORMAT	CV_FOURCC('M', 'J', 'P', 'G')
#define NUM_FINGERS	5
#define NUM_DEFECTS	8

#define RED     CV_RGB(255, 0, 0)
#define GREEN   CV_RGB(0, 255, 0)
#define BLUE    CV_RGB(0, 0, 255)
#define YELLOW  CV_RGB(255, 255, 0)
#define PURPLE  CV_RGB(255, 0, 255)
#define GREY    CV_RGB(200, 200, 200)

using namespace std;
using namespace cv;
using namespace cv::gpu;

typedef struct ctx {
	VideoCapture	capture;	/* Capture */
	CvVideoWriter	*writer;	/* File recording handle */

	GpuMat	image;		/* Input image */
	GpuMat	thr_image;	/* After filtering and thresholding */
	GpuMat	temp_image1;	/* Temporary image (1 channel) */
	GpuMat	temp_image3;	/* Temporary image (3 channels) */

	vector<Point>		contour;	/* Hand contour */
	vector<int>		hull;		/* Hand convex hull */

	CvPoint		hand_center;
	CvPoint		*fingers;	/* Detected fingers positions */
	CvPoint		*defects;	/* Convexity defects depth points */

	Mat	kernel;	/* Kernel for morph operations */

	int		num_fingers;
	int		hand_radius;
	int		num_defects;
} ImageInfo;

void init_capture(ImageInfo *ctx)
{
	ctx->capture.open(0);
	if (!ctx->capture.isOpened()) {
		fprintf(stderr, "Error initializing capture\n");
		exit(1);
	}
	
	Mat init_buffer;
	ctx->capture.read(init_buffer);
	ctx->image.upload(init_buffer);
}

void init_recording(ImageInfo *ctx)
{
	int fps, width, height;

	fps = ctx->capture.get(CV_CAP_PROP_FPS);
	width = ctx->capture.get(CV_CAP_PROP_FRAME_WIDTH);
	height = ctx->capture.get(CV_CAP_PROP_FRAME_HEIGHT);

	if (fps < 0)
		fps = 10;

	ctx->writer = cvCreateVideoWriter(VIDEO_FILE, VIDEO_FORMAT, fps,
					  cvSize(width, height), 1);

	if (!ctx->writer) {
		fprintf(stderr, "Error initializing video writer\n");
		exit(1);
	}
}

void init_windows(void)
{
	cvNamedWindow("output", CV_WINDOW_AUTOSIZE);
	cvNamedWindow("thresholded", CV_WINDOW_AUTOSIZE);
	cvMoveWindow("output", 50, 50);
	cvMoveWindow("thresholded", 700, 50);
}

void init_ctx(ImageInfo *ctx)
{
	ctx->thr_image.create(ctx->image.size(), CV_8UC1);
	ctx->temp_image1.create(ctx->image.size(), CV_8UC1);
	ctx->temp_image3.create(ctx->image.size(), CV_8UC3);
	ctx->kernel = getStructuringElement(MORPH_RECT,Size(9, 9), Point(4, 4));
	ctx->contour_st = cvCreateMemStorage(0);
	ctx->hull_st = cvCreateMemStorage(0);
	ctx->temp_st = cvCreateMemStorage(0);
	ctx->fingers = new CvPoint[NUM_FINGERS + 1]();
	ctx->defects = new CvPoint[NUM_DEFECTS]();
}

void filter_and_threshold(ImageInfo *ctx)
{

	/* Soften image */
	GaussianBlur(ctx->image, ctx->temp_image3, Size(11,11), 0, 0);
	/* Remove some impulsive noise */
	medianBlur(ctx->temp_image3, ctx->temp_image3, 11);

	cvtColor(ctx->temp_image3, ctx->temp_image3, CV_BGR2HSV);

	/*
	 * Apply threshold on HSV values to detect skin color
	 */
	inRange(ctx->temp_image3,
		   Scalar(0, 55, 90, 255),
		   Scalar(28, 175, 230, 255),
		   ctx->thr_image);

	/* Apply morphological opening */
	morphologyEx(ctx->thr_image, ctx->thr_image, MORPH_OPEN, ctx->kernel);
	GaussianBlur(ctx->thr_image, ctx->thr_image, Size(3,3), 0, 0);
}

void find_contour(ImageInfo *ctx)
{
	double area, max_area = 0.0;
	//CvSeq *contours, *tmp, *contour = NULL;
	vector<vector<Point> > contours;
	vector<vector<Point> >::iterator contour;

	/* cvFindContours modifies input image, so make a copy */
	ctx->thr_image.copyTo(ctx->temp_image1);
	findContours(ctx->temp_image1, contours, CV_RETR_EXTERNAL,
		       CV_CHAIN_APPROX_SIMPLE, Point(0, 0));

	/* Select contour having greatest area */
	for (vector<vector<Point> >::iterator iter = contours.begin(),
	     stop = contours.end(); iter != stop; ++iter) {
		area = fabs(contourArea(*iter));
		if (area > max_area) {
			max_area = area;
			contour = iter;
		}
	}

	/* Approximate contour with poly-line */
	if (max_area > 0.0) {
		approxPolyDP(*contour, ctx->contour, 2,
				       true);
	}
}

void find_convex_hull(ImageInfo *ctx)
{
	vector<Vec4i> defects;
	CvConvexityDefect *defect_array;
	int i;
	int x = 0, y = 0;
	int dist = 0;

	if (ctx->contour.empty())
		return;

	convexHull(ctx->contour, ctx->hull, true, true);

	if (!ctx->hull.empty()) {

		/* Get convexity defects of contour w.r.t. the convex hull */
		convexityDefects(ctx->contour, ctx->hull,
					     defects);
		if (!defects.empty()) {
			/* Average depth points to get hand center */
			for (i = 0; i < defects.size() && i < NUM_DEFECTS; ++i) {
				x += ctx.contour[defects[i].[2]].x;
				y += ctx.contour[defects[i].[2]].y;
				
				
				//TODO: EVERYTHING BELOW THIS

				ctx->defects[i] = cvPoint(defect_array[i].depth_point->x,
							  defect_array[i].depth_point->y);
			}

			x /= defects->total;
			y /= defects->total;

			ctx->num_defects = defects->total;
			ctx->hand_center = cvPoint(x, y);

			/* Compute hand radius as mean of distances of
			   defects' depth point to hand center */
			for (i = 0; i < defects->total; i++) {
				int d = (x - defect_array[i].depth_point->x) *
					(x - defect_array[i].depth_point->x) +
					(y - defect_array[i].depth_point->y) *
					(y - defect_array[i].depth_point->y);

				dist += sqrt(d);
			}

			ctx->hand_radius = dist / defects->total;
			free(defect_array);
		}
	}
}

void find_fingers(ImageInfo *ctx)
{
	int n;
	int i;
	CvPoint *points;
	CvPoint max_point;
	int dist1 = 0, dist2 = 0;

	ctx->num_fingers = 0;

	if (!ctx->contour || !ctx->hull)
		return;

	n = ctx->contour->total;
	points = (CvPoint*) calloc(n, sizeof(CvPoint));

	cvCvtSeqToArray(ctx->contour, points, CV_WHOLE_SEQ);

	/*
	 * Fingers are detected as points where the distance to the center
	 * is a local maximum
	 */
	for (i = 0; i < n; i++) {
		int dist;
		int cx = ctx->hand_center.x;
		int cy = ctx->hand_center.y;

		dist = (cx - points[i].x) * (cx - points[i].x) +
		    (cy - points[i].y) * (cy - points[i].y);

		if (dist < dist1 && dist1 > dist2 && max_point.x != 0
		    && max_point.y < cvGetSize(ctx->image).height - 10) {

			ctx->fingers[ctx->num_fingers++] = max_point;
			if (ctx->num_fingers >= NUM_FINGERS + 1)
				break;
		}

		dist2 = dist1;
		dist1 = dist;
		max_point = points[i];
	}

	free(points);
}

void display(ImageInfo *ctx)
{
	int i;

	if (ctx->num_fingers == NUM_FINGERS) {

#if defined(SHOW_HAND_CONTOUR)
		cvDrawContours(ctx->image, ctx->contour, BLUE, GREEN, 0, 1,
			       CV_AA, cvPoint(0, 0));
#endif
		cvCircle(ctx->image, ctx->hand_center, 5, PURPLE, 1, CV_AA, 0);
		cvCircle(ctx->image, ctx->hand_center, ctx->hand_radius,
			 RED, 1, CV_AA, 0);

		for (i = 0; i < ctx->num_fingers; i++) {

			cvCircle(ctx->image, ctx->fingers[i], 10,
				 GREEN, 3, CV_AA, 0);

			cvLine(ctx->image, ctx->hand_center, ctx->fingers[i],
			       YELLOW, 1, CV_AA, 0);
		}

		for (i = 0; i < ctx->num_defects; i++) {
			cvCircle(ctx->image, ctx->defects[i], 2,
				 GREY, 2, CV_AA, 0);
		}
	}

	cvShowImage("output", ctx->image);
	cvShowImage("thresholded", ctx->thr_image);
}

int main(int argc, char **argv)
{
	ImageInfo ctx = { };
	int key;

	init_capture(&ctx);
	init_recording(&ctx);
	init_windows();
	init_ctx(&ctx);

	do {
		ctx.image = cvQueryFrame(ctx.capture);

		filter_and_threshold(&ctx);
		find_contour(&ctx);
		find_convex_hull(&ctx);
		find_fingers(&ctx);

		display(&ctx);
		cvWriteFrame(ctx.writer, ctx.image);

		key = cvWaitKey(1);
	} while (key != 'q');

	return 0;
}
