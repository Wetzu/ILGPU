using System.Data;
using ILGPU;
using ILGPU.Runtime;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

public class Mandelbrot
{
    static void Calculate(Index2D index, ArrayView2D<byte, Stride2D.DenseX> output, float xMin, float xMax, float yMin, float yMax)
    {
        float x0 = xMin + index.X * (xMax - xMin) / 800;
        float y0 = yMin + index.Y * (yMax - yMin) / 600;
        float x = 0;
        float y = 0;
        int iteration = 0;
        int maxIteration = 1000;
        while (x * x + y * y < 2 * 2 && iteration < maxIteration)
        {
            float xtemp = x * x - y * y + x0;
            y = 2 * x * y + y0;
            x = xtemp;
            iteration++;
        }
        if (iteration < maxIteration)
        {
            output[index] = (byte)(255 - (iteration * 255) / maxIteration);
        }
        else
        {
            output[index] = 0;
        }
    }

    public static void Main()
    {
        var image = new Image<Rgba32>(800, 600);

        using Context context = Context.CreateDefault();
        var device = context.GetPreferredDevice(false);

        byte[,] imageData = new byte[800,600];

        using (var accelerator = device.CreateAccelerator(context))
        {
            var intensity = accelerator.Allocate2DDenseX<byte>(new(800, 600));

            var kernel =
                accelerator
                    .LoadAutoGroupedKernel<Index2D, ArrayView2D<byte, Stride2D.DenseX>, float, float, float, float>(
                        Calculate);

            kernel(accelerator.DefaultStream, new Index2D(800, 600), intensity.View, -2, 1, -1, 1);

            imageData = intensity.GetAsArray2D();
        }

        for (int x = 0; x < 800; x++)
        {
            for (int y = 0; y < 600; y++)
            {
                image[x, y] = new Rgba32(imageData[x, y], 0, 0);
            }
        }

        image.Save("output.bmp");
    }
}