## Source Code Walkthrough

### Convolution Operation

How to calculate a 2-dimentional feature map in R, given a 2-dimensional input image with a 2-dimensional filter?

Below illustration will be implemented with a source code snippet in R.
![convolution_operation](./pix/convolution_operation.png) 


Step 1: Initialise a 2-dimensional input image:
```
> input_image <- matrix(c(3,2,2,0,6,4,1,2,1), nrow=3, ncol=3)
```

Verify the matrix of the input image:
```
> input_image
     [,1] [,2] [,3]
[1,]    3    0    1
[2,]    2    6    2
[3,]    2    4    1
```

Step 2: Initialise a 2-dimensional filter:
```
> filter <- matrix(c(-1,-2,-1,0,0,0,1,2,1), nrow=3, ncol=3)
```

Verify the matrix of the filter:
```
> filter
     [,1] [,2] [,3]
[1,]   -1    0    1
[2,]   -2    0    2
[3,]   -1    0    1
```

Step 3: Calculate the element-wise multiplication (NOT matrix multiplication) of input_image and filter, and finally pass the sum of the output to the feature map:
```
> output <- input_image * filter
```

Verify the output of the element-wise multiplication:
```
> output
     [,1] [,2] [,3]
[1,]   -3    0    1
[2,]   -4    0    4
[3,]   -2    0    1
```

Calculate the sum value of the output:
```
> sum <- sum(output)

> sum
[1] -3
```

Finally -3 will be passed to the feature map in the matrix position (1,1):
```
> feature_map <- array(0, c(3, 3, 1))

> feature_map
, , 1

     [,1] [,2] [,3]
[1,]    0    0    0
[2,]    0    0    0
[3,]    0    0    0

> feature_map[1,1,1] <- sum

> feature_map
, , 1

     [,1] [,2] [,3]
[1,]   -3    0    0
[2,]    0    0    0
[3,]    0    0    0
```

### How to debug in RStudio

Step 1: Move the curser to the line in the source code, and choose "Debug" -> "Toggle Breakpoint"
![toggle_breakpoint](./pix/toggle_breakpoint.png)

The red breakpoint dot will then display alongside the line you want to debug for.
![breakpoint](./pix/breakpoint.png)

Finally just run the source code as usual, and it will stop at the very line for you to debug for the temporary output.
![output](./pix/output.png)

You can proceed line by line by clicking "Next" button at the top of the console.

### References:

* https://www.statmethods.net/advstats/matrix.html
* http://www.cookbook-r.com/Numbers/Generating_random_numbers/