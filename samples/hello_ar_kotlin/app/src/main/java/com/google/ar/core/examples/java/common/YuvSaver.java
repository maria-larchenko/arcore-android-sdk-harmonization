package com.google.ar.core.examples.java.common;

import android.media.Image;
import android.graphics.ImageFormat;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;

/** Utility class to dump an Image in {@link ImageFormat#YUV_420_888} to a raw *.yuv file. */
public final class YuvSaver {

    private YuvSaver() {
        // Utility
    }

    /**
     * Writes the three Y, U and V planes of the given {@link Image} sequentially into {@code out}.
     * The output file will therefore contain width*height (Y) + width*height/4 (U) + width*height/4 (V)
     * bytes.
     */
    public static void save(Image image, File out) throws IOException {
        if (image.getFormat() != ImageFormat.YUV_420_888) {
            throw new IllegalArgumentException("Unsupported format: " + image.getFormat());
        }
        FileOutputStream fos = new FileOutputStream(out);
        try {
            Image.Plane[] planes = image.getPlanes();
            for (Image.Plane plane : planes) {
                ByteBuffer buffer = plane.getBuffer();
                buffer.rewind();
                byte[] data = new byte[buffer.remaining()];
                buffer.get(data);
                fos.write(data);
            }
        } finally {
            fos.close();
        }
    }
} 