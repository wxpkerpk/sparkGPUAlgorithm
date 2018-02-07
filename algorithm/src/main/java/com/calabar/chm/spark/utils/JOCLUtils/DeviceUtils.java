package com.calabar.chm.spark.utils.JOCLUtils;

import org.jocl.Pointer;
import org.jocl.Sizeof;
import org.jocl.cl_device_id;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;

import static org.jocl.CL.CL_DEVICE_GLOBAL_MEM_SIZE;
import static org.jocl.CL.CL_DEVICE_MAX_MEM_ALLOC_SIZE;
import static org.jocl.CL.clGetDeviceInfo;

/**
 * Created by wx on 2017/3/17.
 */
public class DeviceUtils {
    public static long getSize(cl_device_id device, int paramName) {
        return getSizes(device, paramName, 1)[0];
    }

    public static long getLong(cl_device_id device, int paramName) {
        return getLongs(device, paramName, 1)[0];
    }

    public static long[] getLongs(cl_device_id device, int paramName, int numValues) {
        long values[] = new long[numValues];
        clGetDeviceInfo(device, paramName, Sizeof.cl_long * numValues, Pointer.to(values), null);
        return values;
    }


    public static long caculateSize(Object... values) {

        long sum = 0;
        for (Object o : values) {
            if (o instanceof float[]) {
                sum += ((float[]) o).length * 4;
            } else if (o instanceof Double[]) {
                sum += ((Double[]) o).length * 8;
            } else if (o instanceof int[]) {
                sum += ((int[]) o).length * 4;
            }

        }
        return sum;
    }

    public static int getGcd(int i, int j) {
        int k;
        while ((k = i % j) != 0) {
            i = j;
            j = k;
        }
        return j;
    }

    public static int getWorkDims(int dataLen, int maxItem) {

        int gcd = getGcd(dataLen, maxItem);
        while (gcd > 65) {
            gcd /= 2;
        }
        return gcd;


    }

    public static long getMaxAllocSize(cl_device_id device) {

        long[] longs = new long[1];

        clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_SIZE, 8, Pointer.to(longs), null);

        return longs[0];


    }

    public static long[] getSizes(cl_device_id device, int paramName, int numValues) {
        // The size of the returned data has to depend on
        // the size of a size_t, which is handled here
        ByteBuffer buffer = ByteBuffer.allocate(
                numValues * Sizeof.size_t).order(ByteOrder.nativeOrder());
        clGetDeviceInfo(device, paramName, Sizeof.size_t * numValues,
                Pointer.to(buffer), null);
        long values[] = new long[numValues];
        if (Sizeof.size_t == 4) {
            for (int i = 0; i < numValues; i++) {
                values[i] = buffer.getInt(i * Sizeof.size_t);
            }
        } else {
            for (int i = 0; i < numValues; i++) {
                values[i] = buffer.getLong(i * Sizeof.size_t);
            }
        }
        return values;
    }
}
