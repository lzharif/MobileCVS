package com.luzharif.mobilecvs;

import org.opencv.core.Algorithm;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.ml.ANN_MLP;
import org.opencv.ml.StatModel;

/**
 * Created by LuZharif on 17/04/2016.
 */
public class Klasifikasi {
    public Mat KlasifikasiBuah(Mat dataFitur) {
        Mat dataHasil = new Mat();
        ANN_MLP annMlp = ANN_MLP.create();
        return dataHasil;

    }

}
