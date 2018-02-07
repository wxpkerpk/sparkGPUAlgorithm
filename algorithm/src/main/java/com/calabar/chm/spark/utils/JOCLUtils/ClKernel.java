package com.calabar.chm.spark.utils.JOCLUtils;

import org.jocl.*;

import java.util.Arrays;

import static org.jocl.CL.*;
import static org.jocl.CL.clCreateKernel;

/**
 * Created by wx on 2017/3/27.
 */
public class ClKernel {
    static  volatile   Long totalTime=0l;
     String functionName;
     String programSource ;
    /**
     * The OpenCL context
     */
    public cl_context context;

    /**
     * The OpenCL command queue to which the all work will be dispatched
     */
    public cl_command_queue commandQueue;

    /**
     * The OpenCL program containing the reduction kernel
     */
    public cl_program program;

    /**
     * The OpenCL kernel that performs the reduction
     */
    public cl_kernel kernel;
    private   long maxItemsSize;




    public cl_device_id device;
    private  volatile boolean init= false;
    public ClKernel(String functionName,String programSource) {
        this.functionName=functionName;
        this.programSource=programSource;
        initialize();
    }
    public  void initialize() {

        if(init){
            return;
        }
        // The platform, device type and device number
        // that will be used
        final int platformIndex = 0;
        final long deviceType = CL_DEVICE_TYPE_ALL;
         int deviceIndex = 0;

        // Enable exceptions and subsequently omit error checks in this sample
        CL.setExceptionsEnabled(true);

        // Obtain the number of platforms
        int numPlatformsArray[] = new int[1];
        System.out.println("================================1");
        clGetPlatformIDs(0, null, numPlatformsArray);
        System.out.println("================================2");

        int numPlatforms = numPlatformsArray[0];



        // Obtain a platform ID
        cl_platform_id platforms[] = new cl_platform_id[numPlatforms];
        clGetPlatformIDs(platforms.length, platforms, null);


        cl_platform_id platform = platforms[platformIndex];
//
//        for(cl_platform_id p:platforms)
//        {
//
//            String platformName=getString(p,CL_PLATFORM_NAME);
//            System.out.println(platformName);
//
//        }
        // Initialize the context properties
        cl_context_properties contextProperties = new cl_context_properties();
        contextProperties.addProperty(CL_CONTEXT_PLATFORM, platform);

        // Obtain the number of devices for the platform
        int numDevicesArray[] = new int[1];
        clGetDeviceIDs(platform, deviceType, 0, null, numDevicesArray);
        int numDevices = numDevicesArray[0];

        // Obtain a device ID
        cl_device_id devices[] = new cl_device_id[numDevices];
        clGetDeviceIDs(platform, deviceType, numDevices, devices, null);
        for (int i=0; i<numDevices; i++)
        {
            String deviceName = getString(devices[i], CL_DEVICE_NAME).toLowerCase();
            if(deviceName.contains("hd")) deviceIndex=i;
            if(deviceName.contains("amd")||deviceName.contains("nvidia")){ deviceIndex=i;break;}
        }
         device = devices[deviceIndex];

        //-------------------------------------
        //读取当前使用设备名称
        byte[] temp = new byte[64];
        long[] size = new long[1];
        CL.clGetDeviceInfo(device, CL.CL_DEVICE_NAME, 64, Pointer.to(temp), size);
        System.out.println("devicename="+new String(temp,0,((int)size[0]-1)));

        ///-----------------------------------

        // Create a context for the selected device
        context = clCreateContext(contextProperties, 1, new cl_device_id[] { device }, null, null, null);

        // Create a command-queue for the selected device
        commandQueue = clCreateCommandQueue(context, device, 0, null);

        // Create the program from the source code
        program = clCreateProgramWithSource(context, 1, new String[] { programSource }, null, null);

        // Build the program
        Object c=clBuildProgram(program, 0, null, null, null, null);

        // Create the kernel
        kernel = clCreateKernel(program, functionName, null);

        init = true;
    }





        /**
         * Returns the value of the platform info parameter with the given name
         *
         * @param platform The platform
         * @param paramName The parameter name
         * @return The value
         */
        private static String getString(cl_platform_id platform, int paramName)
        {
            long size[] = new long[1];
            clGetPlatformInfo(platform, paramName, 0, null, size);
            byte buffer[] = new byte[(int)size[0]];
            clGetPlatformInfo(platform, paramName,
                    buffer.length, Pointer.to(buffer), null);
            return new String(buffer, 0, buffer.length-1);
        }

        /**
         * Returns the value of the device info parameter with the given name
         *
         * @param device The device
         * @param paramName The parameter name
         * @return The value
         */
        private static String getString(cl_device_id device, int paramName)
        {
            long size[] = new long[1];
            clGetDeviceInfo(device, paramName, 0, null, size);
            byte buffer[] = new byte[(int)size[0]];
            clGetDeviceInfo(device, paramName,
                    buffer.length, Pointer.to(buffer), null);
            return new String(buffer, 0, buffer.length-1);
        }







}
