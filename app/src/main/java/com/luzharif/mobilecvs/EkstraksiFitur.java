package com.luzharif.mobilecvs;

import org.opencv.android.OpenCVLoader;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfDouble;
import org.opencv.core.MatOfFloat;
import org.opencv.core.MatOfInt;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.imgproc.Moments;

import java.util.ArrayList;
import java.util.List;
import java.util.Vector;

/**
 * Created by LuZharif on 17/04/2016.
 */
public class EkstraksiFitur {
    float entropi = 0, energi = 0, kontras = 0, homogenitas = 0;

    public Mat EkstrakFiturBuah(Mat citraBuah) {
        if (citraBuah.empty())
            return null;

        //Inisialisasi
        int luasBuah = 0, luasConvexBuah = 0, kelilingBuah = 0;
        float rerataL = 0, rerataA = 0, rerataB = 0, rerata = 0, nilaiStdDevBuah = 0;
        float[] temp = new float[13];
        Mat dataFitur = new Mat(13, 1, CvType.CV_32F);
        Mat citraPerbaikan = new Mat();
        Mat citraAbu = new Mat();
        Mat citraLabAwal = new Mat();
        Mat citraV = new Mat();
        Mat ambangBuahHasil = new Mat();
        Mat konturTemp = new Mat();
        Mat hierarkiKonturBuah = new Mat();
        Mat citraKonturBuah = new Mat();
        Mat citraHullBuah = new Mat();
        Mat citraAreaHullBuah = new Mat();
        Mat citraModifL = new Mat();
        Mat citraModifA = new Mat();
        Mat citraModifB = new Mat();
        Mat citraModifikasiBuah = new Mat();
        Mat citraPaduAbu = new Mat();
        List<Mat> citraLab = new ArrayList<>();
        List<Mat> citraChannel = new ArrayList<>();
        List<MatOfPoint> konturBuah = new ArrayList<>();
        List<MatOfInt> hullBuah = new ArrayList<>();
        Point[] titikHullBuah = new Point[200];
        MatOfDouble rataBuahD = new MatOfDouble();
        MatOfDouble sdBuahD = new MatOfDouble();
        MatOfDouble rataBuahLD = new MatOfDouble();
        MatOfDouble rataBuahAD = new MatOfDouble();
        MatOfDouble rataBuahBD = new MatOfDouble();
        MatOfDouble sdBuahLD = new MatOfDouble();
        MatOfDouble sdBuahAD = new MatOfDouble();
        MatOfDouble sdBuahBD = new MatOfDouble();

        Imgproc.resize(citraBuah, citraBuah, new Size(640, 480), 0, 0, Imgproc.INTER_CUBIC);
        Imgproc.cvtColor(citraBuah, citraAbu, Imgproc.COLOR_BGR2GRAY);
        Imgproc.cvtColor(citraBuah, citraPerbaikan, Imgproc.COLOR_BGR2HSV);
        Imgproc.cvtColor(citraBuah, citraLabAwal, Imgproc.COLOR_BGR2Lab);
        Core.split(citraLabAwal, citraLab);
        Core.split(citraPerbaikan, citraChannel);
        citraV = citraChannel.get(2);

        Imgproc.threshold(citraAbu, ambangBuahHasil, 128, 255, Imgproc.THRESH_OTSU);

        konturTemp = ambangBuahHasil.clone();
        Imgproc.findContours(ambangBuahHasil, konturBuah, hierarkiKonturBuah, Imgproc.RETR_TREE,
                Imgproc.CHAIN_APPROX_SIMPLE, new Point(0, 0));
        citraKonturBuah = Mat.zeros(ambangBuahHasil.rows(), ambangBuahHasil.cols(), CvType.CV_8UC3);
        citraHullBuah = citraKonturBuah.clone();
        citraAreaHullBuah = citraKonturBuah.clone();
        for (int i = 0; i < konturBuah.size(); i++)
            hullBuah.add(new MatOfInt());
        for (int i = 0; i < konturBuah.size(); i++)
            Imgproc.convexHull(konturBuah.get(i), hullBuah.get(i));

        // Loop over all contours
        List<Point[]> hullpoints = new ArrayList<Point[]>();
        for (int i = 0; i < hullBuah.size(); i++) {
            Point[] points = new Point[hullBuah.get(i).rows()];

            // Loop over all points that need to be hulled in current contour
            for (int j = 0; j < hullBuah.get(i).rows(); j++) {
                int index = (int) hullBuah.get(i).get(j, 0)[0];
                points[j] = new Point(konturBuah.get(i).get(index, 0)[0], konturBuah.get(i).get(index, 0)[1]);
            }

            hullpoints.add(points);
        }

        // Convert Point arrays into MatOfPoint
        List<MatOfPoint> hullmop = new ArrayList<MatOfPoint>();
        for (int i = 0; i < hullpoints.size(); i++) {
            MatOfPoint mop = new MatOfPoint();
            mop.fromArray(hullpoints.get(i));
            hullmop.add(mop);
        }

        // Draw contours + hull results
        Scalar color = new Scalar(255, 255, 255);   //Putih
        for (int i = 0; i < konturBuah.size(); i++) {
            Imgproc.drawContours(citraKonturBuah, konturBuah, i, color); //TODO cek apakah drawContours sudah benar
            Imgproc.drawContours(citraKonturBuah, hullmop, i, color);
        }
        Imgproc.fillPoly(citraAreaHullBuah, konturBuah, color); //TODO cek apakah fillPoly sudah benar

        luasConvexBuah = LuasConvex(citraAreaHullBuah);
        Core.bitwise_and(ambangBuahHasil, citraLab.get(0), citraModifL);
        Core.bitwise_and(ambangBuahHasil, citraLab.get(1), citraModifA);
        Core.bitwise_and(ambangBuahHasil, citraLab.get(2), citraModifB);
        Core.bitwise_and(ambangBuahHasil, citraV, citraModifikasiBuah);
        Core.meanStdDev(citraModifikasiBuah, rataBuahD, sdBuahD);
        Core.meanStdDev(citraModifL, rataBuahLD, sdBuahLD);
        Core.meanStdDev(citraModifA, rataBuahAD, sdBuahAD);
        Core.meanStdDev(citraModifB, rataBuahBD, sdBuahBD);
        rerataL = (float) rataBuahLD.get(0, 0)[0];
        rerataA = (float) rataBuahAD.get(0, 0)[0];
        rerataB = (float) rataBuahBD.get(0, 0)[0];
        rerata = (float) rataBuahD.get(0, 0)[0];
        nilaiStdDevBuah = (float) sdBuahD.get(0, 0)[0];
        luasBuah = Core.countNonZero(ambangBuahHasil);
        kelilingBuah = Keliling(citraKonturBuah);

        Core.bitwise_and(ambangBuahHasil, citraAbu, citraPaduAbu);
        Tekstur(citraPaduAbu);

        //Input dataFitur
        temp[0] = luasBuah;
        temp[1] = kelilingBuah;
        temp[2] = rerataL;
        temp[3] = rerataA;
        temp[4] = rerataB;
        temp[5] = (float) luasBuah / luasConvexBuah;
        temp[6] = nilaiStdDevBuah;
        temp[7] = CircularityBuah(luasBuah, kelilingBuah);
        temp[8] = EccentricityBuah(konturBuah);
        temp[9] = entropi;
        temp[10] = energi;
        temp[11] = kontras;
        temp[12] = homogenitas;

        for(int i = 0; i < 13; i++)
            dataFitur.put(i, 0, temp[i]);
        return dataFitur;
    }

    private void Tekstur(Mat citraPaduAbu) {
        double energy = 0, contrast = 0, homogenity = 0, IDM = 0, entropy = 0, mean1 = 0;
        //array <float, 6> tekstur;
        int row = citraPaduAbu.rows(), col = citraPaduAbu.cols();
        Mat gl = Mat.zeros(256, 256, CvType.CV_32FC1);

        //creating glcm matrix with 256 levels,radius=1 and in the horizontal direction
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < col - 1; j++) {
                int xBaru = (int) citraPaduAbu.get(i, j)[0];
                int yBaru = (int) citraPaduAbu.get(i, j + 1)[0];
                double baru = (int) gl.get(xBaru, yBaru)[0] + 1;
                gl.put(xBaru, yBaru, baru);
            }
        }
        // normalizing glcm matrix for parameter determination
        Core.add(gl, gl.t(), gl);
        Core.divide(gl, Core.sumElems(gl), gl);

        for (int i = 0; i < 256; i++)
            for (int j = 0; j < 256; j++) {
                double nilaiGl = gl.get(i, j)[0];
                energy = energy + (nilaiGl * nilaiGl);            //finding parameters
                contrast = contrast + (i - j) * (i - j) * nilaiGl;
                homogenity = homogenity + nilaiGl / (1 + Math.abs(i - j));
                if (nilaiGl != 0)
                    entropy = entropy - nilaiGl * Math.log10((nilaiGl));
            }
        entropi = (float) entropy;
        energi = (float) energy;
        kontras = (float) contrast;
        homogenitas = (float) homogenity;
    }

    private float EccentricityBuah(List<MatOfPoint> konturBuah) {
        List<Moments> mu = new ArrayList<>(konturBuah.size());
        int largestContourIndex = 0;
        double largestArea = 0;
        float myu20 = 0, myu11 = 0, myu02 = 0, eigenValue1 = 0, eigenValue2 = 0, eccentricityBuah = 0;
        float[] myu = new float[3];
        Mat eigenV = new Mat();
        Mat eigenVct = new Mat();
        MatOfFloat matriks = new MatOfFloat(2, 2);

        for (int i = 0; i < konturBuah.size(); i++) {
            double area = Imgproc.contourArea(konturBuah.get(i), false);  //  Find the area of contour
            if (area > largestArea) {
                largestArea = area;
                largestContourIndex = i;
            }
        }
        mu.add(0, Imgproc.moments(konturBuah.get(largestContourIndex), false));
        myu[0] = (float) mu.get(0).get_m20();
        myu[1] = (float) mu.get(0).get_m11();
        myu[2] = (float) mu.get(0).get_m02();

        //Input nilai matriks ke dalam variabel
        //[myu20	myu11]
        //[myu11	myu02]
        matriks.put(0, 0, myu[0]);
        matriks.put(1, 0, myu[1]);
        matriks.put(0, 1, myu[1]);
        matriks.put(1, 1, myu[2]);

        //Hitung nilai eigen
        Core.eigen(matriks, eigenV, eigenVct);
        eigenValue1 = (float) eigenVct.get(0, 0)[0];
        eigenValue2 = (float) eigenVct.get(1, 0)[0];

        //Perhitungan eccentricity
        if (eigenValue1 >= eigenValue2)
            eccentricityBuah = eigenValue2 / eigenValue1;
        else
            eccentricityBuah = eigenValue1 / eigenValue2;

        return eccentricityBuah;
    }

    private float CircularityBuah(int luasBuah, int kelilingBuah) {
        float circularity = (float) ((float) 4 * Math.PI * luasBuah / Math.pow(kelilingBuah, 2));
        return circularity;
    }

    private int LuasConvex(Mat citraAreaHullBuah) {
        int luas = 0;
        int barisCitra = citraAreaHullBuah.rows();
        int kolomCitra = citraAreaHullBuah.cols();
        for (int x = 0; x < barisCitra; x++) {
            for (int y = 0; y < kolomCitra; y++) {
                if (citraAreaHullBuah.get(x, y)[0] > 0) {
                    luas++;
                }
            }
        }
        return luas;
    }

    private int Keliling(Mat citraKonturBuah) {
        int keliling = 0;
        int barisCitra = citraKonturBuah.rows();
        int kolomCitra = citraKonturBuah.cols();
        for (int x = 0; x < barisCitra; x++) {
            for (int y = 0; y < kolomCitra; y++) {
                if (citraKonturBuah.get(x, y)[0] > 0) {
                    keliling++;
                }
            }
        }
        return keliling;
    }

}
