

using System.Diagnostics;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Running;
using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.CPU;
using ILGPU.Runtime.Cuda;

public class ILGPUVectorAdd
{
    static readonly Random LocalRandom = new Random();
    private const int ArrayLength = 1000000;
    private const int MaxRandomValue = Int16.MaxValue;
    private const int PrintLines = 10;

    static void VectorAdd(Index1D index, ArrayView<int> a, ArrayView<int> b, ArrayView<int> c)
    {
        c[index] = a[index] * b[index];
    }


    static void InitializeArray(int[] array, bool random = true)
    {
        for (int i = 0; i < array.Length; i++)
        {
            if (random)
            {
                array[i] = LocalRandom.Next(MaxRandomValue);
            }
            else
            {
                array[i] = 0;
            }
        }
    }

    public static void Main()
    {

        //Initializing Context, Device and generating Accelerator
        using Context context = Context.CreateDefault();
        long cpuTicks;
        long gpuTicks;
        
        var gpuDevice = context.GetCudaDevice(0);


        //Preparing Test-Data
        var aArray = new int[ArrayLength];
        var bArray = new int[ArrayLength];
        var cArray = new int[ArrayLength];

        //Initialize Test-Data
        InitializeArray(aArray);
        InitializeArray(bArray);
        InitializeArray(cArray, false);

        using (var accelerator = gpuDevice.CreateCudaAccelerator(context))
        {
            var stopWatch = new Stopwatch();
            stopWatch.Start();

            //Allocating Memory on Accelerator
            var a = accelerator.Allocate1D<int>(aArray);
            var b = accelerator.Allocate1D<int>(bArray);
            var c = accelerator.Allocate1D<int>(cArray);


            //Loading and Compiling Kernels
            Action<Index1D, ArrayView<int>, ArrayView<int>, ArrayView<int>> vectorAddKernel =
                accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>, ArrayView<int>, ArrayView<int>>(VectorAdd);

            vectorAddKernel((int)c.Length, a.View, b.View, c.View);

            //Synchronizing
            accelerator.Synchronize();

            stopWatch.Stop();

            var gpuTime = stopWatch.Elapsed;
            gpuTicks = stopWatch.ElapsedTicks;


            Console.WriteLine($"GPU Run took {gpuTime.Seconds}S {gpuTime.Milliseconds}MS {gpuTime.Ticks}Ticks");

            stopWatch.Reset();

            int[] hostOutput = c.GetAsArray1D();

            for (int i = 0; i < PrintLines; i++)
            {
                Console.WriteLine($"A: {aArray[i], 9} B: {bArray[i], 9} Result: {hostOutput[i], 12}");
            }
        }


        var cpuDevice = context.GetCPUDevice(0);

        using (var accelerator = cpuDevice.CreateCPUAccelerator(context))
        {
            var stopWatch = new Stopwatch();
            stopWatch.Start();

            //Allocating Memory on Accelerator
            var a = accelerator.Allocate1D<int>(aArray);
            var b = accelerator.Allocate1D<int>(bArray);
            var c = accelerator.Allocate1D<int>(cArray);


            //Loading and Compiling Kernels
            Action<Index1D, ArrayView<int>, ArrayView<int>, ArrayView<int>> vectorAddKernel =
                accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>, ArrayView<int>, ArrayView<int>>(VectorAdd);

            vectorAddKernel((int)c.Length, a.View, b.View, c.View);

            //Synchronizing
            accelerator.Synchronize();

            stopWatch.Stop();

            var cpuTime = stopWatch.Elapsed;
            cpuTicks = stopWatch.ElapsedTicks;

            Console.WriteLine($"CPU Run took {cpuTime.Seconds}S {cpuTime.Milliseconds}MS {cpuTime.Ticks}Ticks");

            stopWatch.Reset();

            int[] hostOutput = c.GetAsArray1D();

            for (int i = 0; i < PrintLines; i++)
            {
                Console.WriteLine($"A: {aArray[i],9} B: {bArray[i],9} Result: {hostOutput[i], 12}");
            }
        }

        // moved output data from the GPU to the CPU for output to console

        if (gpuTicks < cpuTicks)
        {
            Console.WriteLine($"GPU was faster than CPU");
        }
        else
        {
            Console.WriteLine($"CPU was faster than GPU");
        }

        //var summary = BenchmarkRunner.Run(typeof(VectorAddBenchmark));
    }
}

public class VectorAddBenchmark
{

    static readonly Random LocalRandom = new Random();
    private const int ArrayLength = 1000000;
    private const int MaxRandomValue = Int16.MaxValue;

    private Context Context { get; set; }

    private CudaDevice CudaD { get; set; }

    private CPUDevice CpuD { get; set; }

    private int[] AData { get; set; }

    private int[] BData { get; set; }

    private int[] CData { get; set; }


    static void VectorAddKernel(Index1D index, ArrayView<int> a, ArrayView<int> b, ArrayView<int> c)
    {
        c[index] = a[index] * b[index];
    }


    static void InitializeArray(int[] array, bool random = true)
    {
        for (int i = 0; i < array.Length; i++)
        {
            if (random)
            {
                array[i] = LocalRandom.Next(MaxRandomValue);
            }
            else
            {
                array[i] = 0;
            }
        }
    }

    [GlobalSetup]
    public void GlobalSetup()
    {
        Context = Context.CreateDefault();
        CudaD = Context.GetCudaDevice(0);
        CpuD = Context.GetCPUDevice(0);

        AData = new int[ArrayLength];
        BData = new int[ArrayLength];
        CData = new int[ArrayLength];

        InitializeArray(AData);
        InitializeArray(BData);
        InitializeArray(CData, false);
    }

    [GlobalCleanup]
    public void GlobalCleanup()
    {
        Context.Dispose();
    }

    [Benchmark()]
    public void AddOnGPU()
    {
        using (var accelerator = CudaD.CreateCudaAccelerator(Context))
        {
            var a = accelerator.Allocate1D<int>(AData);
            var b = accelerator.Allocate1D<int>(BData);
            var c = accelerator.Allocate1D<int>(CData);


            //Loading and Compiling Kernels
            Action<Index1D, ArrayView<int>, ArrayView<int>, ArrayView<int>> vectorAddKernel =
                accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>, ArrayView<int>, ArrayView<int>>(VectorAddKernel);

            vectorAddKernel((int)c.Length, a.View, b.View, c.View);

            //Synchronizing
            accelerator.Synchronize();
        }
    }

    [Benchmark()]
    public void AddOnCPU()
    {
        using (var accelerator = CpuD.CreateCPUAccelerator(Context))
        {
            var a = accelerator.Allocate1D<int>(AData);
            var b = accelerator.Allocate1D<int>(BData);
            var c = accelerator.Allocate1D<int>(CData);


            //Loading and Compiling Kernels
            Action<Index1D, ArrayView<int>, ArrayView<int>, ArrayView<int>> vectorAddKernel =
                accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>, ArrayView<int>, ArrayView<int>>(VectorAddKernel);

            vectorAddKernel((int)c.Length, a.View, b.View, c.View);

            //Synchronizing
            accelerator.Synchronize();
        }
    }
}