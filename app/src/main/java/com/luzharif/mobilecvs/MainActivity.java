package com.luzharif.mobilecvs;

import android.Manifest;
import android.content.Context;
import android.content.Intent;
import android.content.pm.PackageInfo;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Bundle;
import android.provider.MediaStore;
import android.support.design.widget.FloatingActionButton;
import android.support.design.widget.Snackbar;
import android.support.v7.app.AppCompatActivity;
import android.support.v7.widget.Toolbar;
import android.view.View;
import android.view.Menu;
import android.view.MenuItem;
import android.widget.ImageView;

import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.List;

public class MainActivity extends AppCompatActivity {

    static final int REQUEST_IMAGE_CAPTURE = 1;

    private ImageView citraBuahAsli;
    private ImageView citraBuahL;
    private ImageView citraBuahA;
    private ImageView citraBuahB;

    static {
        if (!OpenCVLoader.initDebug()) {
            // Handle initialization error
        }
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        citraBuahAsli = (ImageView) findViewById(R.id.citrabuah);
        citraBuahL = (ImageView) findViewById(R.id.citrabuahL);
        citraBuahA = (ImageView) findViewById(R.id.citrabuahA);
        citraBuahB = (ImageView) findViewById(R.id.citrabuahB);

        FloatingActionButton fab = (FloatingActionButton) findViewById(R.id.fabambilcitra);
        fab.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                Intent bukaKameraIntent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
                if (bukaKameraIntent.resolveActivity(getPackageManager()) != null) {
                    startActivityForResult(bukaKameraIntent, REQUEST_IMAGE_CAPTURE);
                }
            }
        });
    }

    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        // Inflate the menu; this adds items to the action bar if it is present.
        getMenuInflater().inflate(R.menu.menu_main, menu);
        return true;
    }

    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
        // Handle action bar item clicks here. The action bar will
        // automatically handle clicks on the Home/Up button, so long
        // as you specify a parent activity in AndroidManifest.xml.
        int id = item.getItemId();

        //noinspection SimplifiableIfStatement
        if (id == R.id.action_settings) {
            return true;
        }

        return super.onOptionsItemSelected(item);
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        if (requestCode == REQUEST_IMAGE_CAPTURE && resultCode == RESULT_OK) {
            Bundle extras = data.getExtras();
            Bitmap imageBitmap = (Bitmap) extras.get("data");
            Mat citraOlah = new Mat();
            Mat citraLAB = new Mat();
            List<Mat> citraListLAB = new ArrayList<Mat>();
            Utils.bitmapToMat(imageBitmap, citraOlah);
            Imgproc.resize(citraOlah, citraOlah, new Size(640, 480), 0, 0, Imgproc.INTER_CUBIC);
            List<Mat> channel = new ArrayList<Mat>();

            //Ekualisasi histogram citra
            Mat citraHSV = new Mat();
            Imgproc.cvtColor(citraOlah, citraHSV, Imgproc.COLOR_BGR2HSV);
            Core.split(citraHSV, channel);
            Imgproc.equalizeHist(channel.get(2),channel.get(2));
            Core.merge(channel, citraOlah);

            Imgproc.cvtColor(citraOlah, citraLAB, Imgproc.COLOR_BGR2Lab);
            Core.split(citraLAB, citraListLAB);
            Mat citraL = citraListLAB.get(0);
            Mat citraA = citraListLAB.get(1);
            Mat citraB = citraListLAB.get(2);

            Bitmap citraBitL = Bitmap.createBitmap(citraL.cols(), citraL.rows(), Bitmap.Config.ARGB_8888);
            Bitmap citraBitA = Bitmap.createBitmap(citraA.cols(), citraA.rows(), Bitmap.Config.ARGB_8888);
            Bitmap citraBitB = Bitmap.createBitmap(citraB.cols(), citraB.rows(), Bitmap.Config.ARGB_8888);
            Utils.matToBitmap(citraL, citraBitL);
            Utils.matToBitmap(citraA, citraBitA);
            Utils.matToBitmap(citraB, citraBitB);

            citraBuahAsli.setImageBitmap(imageBitmap);
            citraBuahL.setImageBitmap(citraBitL);
            citraBuahA.setImageBitmap(citraBitA);
            citraBuahB.setImageBitmap(citraBitB);

        }
    }

    public boolean hasPermissionInManifest(Context context, String permissionName) {
        final String packageName = context.getPackageName();
        try {
            final PackageInfo packageInfo = context.getPackageManager()
                    .getPackageInfo(packageName, PackageManager.GET_PERMISSIONS);
            final String[] declaredPermisisons = packageInfo.requestedPermissions;
            if (declaredPermisisons != null && declaredPermisisons.length > 0) {
                for (String p : declaredPermisisons) {
                    if (p.equals(permissionName)) {
                        return true;
                    }
                }
            }
        } catch (PackageManager.NameNotFoundException e) {

        }
        return false;
    }
}
