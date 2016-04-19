package com.luzharif.mobilecvs;

/**
 * Created by LuZharif on 19/04/2016.
 */
public class LinkToNative {
    public native String compare(long src,long dest);

    static {
        System.loadLibrary("compare");
    }
}
