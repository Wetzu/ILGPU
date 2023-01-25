

using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Running;
using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.CPU;
using ILGPU.Runtime.Cuda;

public class ILGPUVectorAdd
{
    static readonly Random LocalRandom = new Random();
    private const int ArrayLength = 256;
    private const int MaxRandomValue = Int16.MaxValue;

    static void VectorAdd(Index1D index, ArrayView<int> a, ArrayView<int> b, ArrayView<int> c)
    {
        c[index] = a[index] + b[index];
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
        var device = context.GetCudaDevice(0);
        using var accelerator = device.CreateCudaAccelerator(context);

        //Preparing Test-Data
        var aArray = new int[ArrayLength];
        var bArray = new int[ArrayLength];
        var cArray = new int[ArrayLength];

        //Initialize Test-Data
        InitializeArray(aArray);
        InitializeArray(bArray);
        InitializeArray(cArray, false);

        //Allocating Memory on Accelerator
        var a = accelerator.Allocate1D<int>(aArray);
        var b = accelerator.Allocate1D<int>(bArray);
        var c = accelerator.Allocate1D<int>(cArray);


        //Loading and Compiling Kernels
        Action<Index1D, ArrayView<int>, ArrayView<int>, ArrayView<int>> vectorAddKernel =
            accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>, ArrayView<int>, ArrayView<int>>(VectorAdd);

        vectorAddKernel((int)c.Length, a.View, b.View, c.View);

        // wait for the accelerator to be finished with whatever it's doing
        // in this case it just waits for the kernel to finish.
        accelerator.Synchronize();

        // moved output data from the GPU to the CPU for output to console
        int[] hostOutput = c.GetAsArray1D();

        for (int i = 0; i < hostOutput.Length; i++)
        {
            Console.WriteLine($"A: {aArray[i]} B: {bArray[i]} Result: {hostOutput[i]}");
        }

        var summary = BenchmarkRunner.Run(typeof(VectorAddBenchmark));
    }
}

public class VectorAddBenchmark
{

    static readonly Random LocalRandom = new Random();
    private const int ArrayLength = 256000;
    private const int MaxRandomValue = Int16.MaxValue;

    private Context Context { get; set; }

    private CudaDevice CudaD { get; set; }

    private CPUDevice CpuD { get; set; }

    private int[] AData { get; set; }

    private int[] BData { get; set; }

    private int[] CData { get; set; }


    static void VectorAddKernel(Index1D index, ArrayView<int> a, ArrayView<int> b, ArrayView<int> c)
    {
        c[index] = a[index] + b[index];
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

            // wait for the accelerator to be finished with whatever it's doing
            // in this case it just waits for the kernel to finish.
            accelerator.Synchronize();
        }
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

            // wait for the accelerator to be finished with whatever it's doing
            // in this case it just waits for the kernel to finish.
            accelerator.Synchronize();
        }
    }
}