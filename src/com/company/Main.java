package com.company;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.awt.image.WritableRaster;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;

import static java.lang.System.currentTimeMillis;


public class Main {

    // Below is the iterative version from
    // https://github.com/eliasyilma/CNN/blob/master/src/cnn/Convolution.java
    public static double[][] m_sub(double[][] mat, int r_s, int r_e, int c_s, int c_e) {
        double[][] sub = new double[r_e - r_s + 1][c_e - c_s + 1];
        for (int i = 0; i < sub.length; i++) {
            for (int j = 0; j < sub[0].length; j++) {
                sub[i][j] = mat[r_s + i][c_s + j];
            }
        }
        return sub;
    }

    public static double mm_elsum(double[][] mat1, double[][] mat2) {
        double sum = 0;
        for (int i = 0; i < mat1.length; i++) {
            for (int j = 0; j < mat2[0].length; j++) {
                sum += mat1[i][j] * mat2[i][j];
            }
        }
        if (sum > 255) {
            sum = 255;
        }
        if (sum < 0) {
            sum = 0;
        }
        return sum;
    }

    public static double[][][] img_to_mat(BufferedImage imageToPixelate) {
        int w = imageToPixelate.getWidth(), h = imageToPixelate.getHeight();
        int[] pixels = imageToPixelate.getRGB (0, 0, w, h, null, 0, w);
        double[][][] dta = new double[4][h][w];

        for (int pixel = 0, row = 0, col = 0; pixel < pixels.length; pixel++) {
            int blue = (pixels[pixel]) & 0x000000FF;
            int green  = (pixels[pixel] >>8 ) & 0x000000FF;
            int red = (pixels[pixel] >> 16) & 0x000000FF;
            int alpha = (pixels[pixel] >> 24) & 0xff;
            dta[0][row][col] = red;
            dta[1][row][col] = green;
            dta[2][row][col] = blue;
            dta[3][row][col] = alpha;
            col++;
            if (col == w) {
                col = 0;
                row++;
            }

        }
        return dta;
    }

    public static double[][] convolve3x3(double[][] image, double[][] filter) {
        double[][] result = new double[image.length - 2][image[0].length - 2];
        //loop through
        for (int i = 1; i < image.length - 1; i++) {
            for (int j = 1; j < image[0].length - 1; j++) {
                double[][] conv_region = m_sub(image, i - 1, i + 1, j - 1, j + 1);
                result[i-1][j-1] = mm_elsum(conv_region, filter);
            }
        }
        return result;
    }

    public static double[][][] convolve3x3_fourChannel(double[][][] image, double[][] filter) {
        double[][][] result = {convolve3x3(image[0],filter),convolve3x3(image[1],filter), convolve3x3(image[2],filter), convolve3x3(image[3],filter)};
        return  result;
    }

    public static BufferedImage mat_to_img(double[][][] image) {
        BufferedImage buf_image = new BufferedImage(image[0][0].length, image[0].length, BufferedImage.TYPE_INT_RGB);
        for (int i = 0; i < image[0].length; i++) {
            for (int j = 0; j < image[0][0].length; j++) {
                int red = (int) Math.round(Math.abs(image[0][i][j]));
                int green = (int) Math.round(Math.abs(image[1][i][j]));
                int blue = (int) Math.round(Math.abs(image[2][i][j]));
                int alpha = (int) Math.round(Math.abs(image[3][i][j]));
                buf_image.setRGB(j, i, (alpha<<24 | red<<16 | green<<8 | blue));
            }
        }
        return buf_image;
    }

    public static void main(String[] args) throws IOException, InterruptedException {

        int num_cores = Runtime.getRuntime().availableProcessors();
        System.out.println(num_cores);


	    // Read in image
        BufferedImage img = null;
        String image_name = "10k1.jpg";
        String format = "jpg";
        img = ImageIO.read(new File("F:\\Study\\UIUC\\Study\\2022FALL\\Parallel Programming\\Nested parallelization of convolutional layer\\src\\com\\company\\" + image_name));
        double[][][] four_channel_img = img_to_mat(img);

        double[][] filter = {{0.0,-1.0,0.0},{-1.0,5.0,-1.0},{0.0,-1.0,0.0}}; // Sharpen
//        double[][] filter = {{-1.0,-1.0,-1.0},{-1.0,8.0,-1.0},{-1.0,-1.0,-1.0}}; // edge detection
//        double[][] filter = {{0.11111111111,0.11111111111,0.11111111111},{0.11111111111,0.11111111111,0.11111111111},{0.11111111111,0.11111111111,0.11111111111}}; // Box blur
//        double[][] filter = {{0.0625,0.125,0.0625},{0.125,0.25,0.125},{0.0625,0.125,0.0625}}; // Gaussian blur
//        double[][] filter = {{0,0,0},{0,1,0},{0,0,0}}; // Identity

//        TODO image needs to split into Red Green Blue Alpha to be conv3x3
//              separately and merged back at the end to get correct conv


        // run on iterative version
        long iter_start =  currentTimeMillis();
        double[][][] filtered_iter_fourChannel = convolve3x3_fourChannel(four_channel_img, filter);
        long iter_end =  currentTimeMillis();
        System.out.println("Iterative takes " + (iter_end - iter_start) + "ms.");
        ImageIO.write(mat_to_img(filtered_iter_fourChannel), format, new File("F:\\Study\\UIUC\\Study\\2022FALL\\Parallel Programming\\Nested parallelization of convolutional layer\\src\\com\\company\\filtered_iter_" + image_name));


        // run on parallel version
        long para_start =  currentTimeMillis();
        double[][][] filtered_parallel_fourChannel = convolutionLayerFourChannel_parallel(four_channel_img, filter);
        long para_end =  currentTimeMillis();
        System.out.println("Parallel takes " + (para_end - para_start) + "ms.");
        ImageIO.write(mat_to_img(filtered_parallel_fourChannel), format, new File("F:\\Study\\UIUC\\Study\\2022FALL\\Parallel Programming\\Nested parallelization of convolutional layer\\src\\com\\company\\filtered_para_" + image_name));

        // compare
        boolean isEqual = java.util.Arrays.deepEquals(filtered_iter_fourChannel,filtered_parallel_fourChannel);
        if (isEqual) {
            System.out.println("Same output.");
        } else {
            System.out.println("Different/Wrong output.");
        }
    }



    // my code is below

    public static class ThreadConv3x3_parallel_strip extends Thread {
        private int[] range;
        private double[][] image;
        private double[][] filter;
        private double[][] result;
        private double[][] tmpResult;
        private int num_rows;
        private int num_cols;
        private int row_shift;
        private int column_shift;

        public ThreadConv3x3_parallel_strip(int[] range, double[][] image, double[][] filter, int row_shift, int column_shift, double[][] result) {
            this.range = range;
            this.image = image;
            this.filter = filter;
            this.row_shift = row_shift;
            this.column_shift = column_shift;
            this.result = result;
            this.tmpResult = new double[range[1] - range[0]][image[0].length - 2];
            this.num_rows = image.length;
            this.num_cols = image[0].length;
        }

        public void run() {
            for (int i = this.range[0]; i < this.range[1]; i++) {
                for (int j = 1; j < (this.num_cols - 1); j++) {
                    this.tmpResult[i - this.range[0]][j - 1] = this.image[i + this.row_shift][j + this.column_shift] * this.filter[1 + this.row_shift][1 + this.column_shift];
                }
            }
            synchronized (this.result) {
                for (int i = 0; i < this.result.length; i++) {
                    for (int j = 0; j < this.result[0].length; j++) {
                        this.result[i][j] = this.result[i][j] + this.tmpResult[i][j];
                    }
                }
            }
        }
    }



    static double[][] conv3x3_parallel_strip(int[] range, double[][] image, double[][] filter) throws InterruptedException {
        // double default set to zero
        double[][] result = new double[range[1] - range[0]][image[0].length - 2];

        ArrayList<Thread> threads = new ArrayList<Thread>();

        for (int row_shift = -1; row_shift <= 1; row_shift++) {
            for (int column_shift = -1; column_shift <= 1; column_shift++) {
                Thread newThread = new ThreadConv3x3_parallel_strip(range, image, filter, row_shift, column_shift, result);
                newThread.start();
                threads.add(newThread);
            }
        }

        for (int i = 0; i < threads.size(); i++) {
            threads.get(i).join();
        }
        return result;
    }


    public static class ThreadStrip extends Thread {
        private int[] range;
        private double[][] image;
        private double[][] filter;
        private double[][] result;
        private double[][] tmpResult;
        private int num_rows;
        private int num_cols;

        public ThreadStrip(int[] range, double[][] image, double[][] filter, double[][] result) {
            this.range = range;
            this.image = image;
            this.filter = filter;
            this.result = result;
            this.tmpResult = new double[range[1] - range[0]][image[0].length - 2];
            this.num_rows = image.length;
            this.num_cols = image[0].length;
        }

        public void run() {
            // iterative version
//            for (int i = this.range[0]; i < this.range[1]; i++) {
//                for (int j = 1; j < (this.num_cols - 1); j++) {
//
//                        // placeholder below
//                        double[][] conv_region = m_sub(this.image, i - 1, i + 1, j - 1, j + 1);
//                        this.tmpResult[i - this.range[0]][j - 1] = mm_elsum(conv_region, this.filter);
//                        // above
//
//                }
//            }

            // parallel version
            try {
                this.tmpResult = conv3x3_parallel_strip(this.range, this.image, this.filter);
            } catch (InterruptedException e) {
                e.printStackTrace();
            }


            synchronized (this.result) {
//                System.arraycopy(this.tmpResult, 0, this.result, this.range[0], range[1] - range[0]);
                for (int i = this.range[0]; i < this.range[1]; i++) {
                    for (int j = 1; j < (this.num_cols - 1); j++) {
                            this.result[i-1][j-1] = this.tmpResult[i - this.range[0]][j - 1];
                            if (this.result[i-1][j-1] > 255) {
                                this.result[i-1][j-1] = 255;
                            }
                            if (this.result[i-1][j-1] < 0) {
                                this.result[i-1][j-1] = 0;
                            }
                    }
                }
            }
        }
    }



    static double[][] conv3x3_parallel(double[][] image, double[][] filter) throws InterruptedException {
        double[][]  result = new double[image.length - 2][image[0].length - 2];
        int num_cores = Runtime.getRuntime().availableProcessors();

        // split image to n parts by indexing
        int interval = (int) Math.floor(((image.length - 1) - 1)/num_cores);
        ArrayList<Thread> threads = new ArrayList<Thread>();


        for (int i = 0; i < num_cores; i++) {
            // [i, i + interval)
            int[] range = { 1 + i*interval, 1 + i*interval + interval};
            Thread newThread = new ThreadStrip(range, image, filter, result);
            newThread.start();
            threads.add(newThread);

        }
        if (num_cores*interval < ((image.length - 1) - 1)) {
            // [1 + num_cores*interval, (image.length - 1) )
            int[] range = {1 + num_cores*interval, (image.length - 1)};
            Thread newThread = new ThreadStrip(range, image, filter, result);
            newThread.start();
            threads.add(newThread);
        }

        for (int i = 0; i < threads.size(); i++) {
            threads.get(i).join();
        }

        return result;
    }

    public static class ThreadConv3x3_parallel extends Thread {
        private double[][] image;
        private double[][] filter;
        private double[][] result;

        public ThreadConv3x3_parallel(double[][] image, double[][] filter, double[][] result) {
            this.image = image;
            this.filter = filter;
            this.result = result;
        }

        public void run() {
            try {
                double[][] tmpResult = conv3x3_parallel(image, filter);
                System.arraycopy(tmpResult, 0, this.result, 0, this.result.length);
            } catch (Exception e) {
                System.out.println(e);
            }
        }
    }


    static double[][][] convolutionLayerFourChannel_parallel(double[][][] image, double[][] filter) throws InterruptedException {
        double[][][] result = new double[4][image[0].length - 2][image[0][0].length - 2];
        ArrayList<Thread> threads = new ArrayList<Thread>();
        for (int i = 0; i < image.length; i++) {
            Thread newChannelThread = new ThreadConv3x3_parallel(image[i], filter, result[i]);
            newChannelThread.run();
            threads.add(newChannelThread);
        }

        for (int i = 0; i < threads.size(); i++) {
            threads.get(i).join();
        }
        return result;
    }
}
