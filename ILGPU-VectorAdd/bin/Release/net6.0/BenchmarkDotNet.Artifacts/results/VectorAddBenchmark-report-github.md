``` ini

BenchmarkDotNet=v0.13.4, OS=Windows 11 (10.0.22000.1455/21H2)
Intel Core i9-9900K CPU 3.60GHz (Coffee Lake), 1 CPU, 16 logical and 8 physical cores
.NET SDK=6.0.405
  [Host]     : .NET 6.0.13 (6.0.1322.58009), X64 RyuJIT AVX2 [AttachedDebugger]
  DefaultJob : .NET 6.0.13 (6.0.1322.58009), X64 RyuJIT AVX2


```
|   Method |      Mean |    Error |   StdDev |
|--------- |----------:|---------:|---------:|
| AddOnGPU | 123.69 ms | 2.396 ms | 3.730 ms |
| AddOnCPU |  98.71 ms | 1.948 ms | 1.822 ms |
