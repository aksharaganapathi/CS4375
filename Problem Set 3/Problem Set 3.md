# Problem 1
1. The VC dimension of the hypothesis space of n-dimensional spheres is $n+1$. In a n-dimensional space, you can shatter any set of $n+1$ points since for any labeling of these points, you can always find a sphere such that all the points labeled +1 are inside the sphere and all the points labeled -1 are outside the sphere or vice versa. For any set of $n + 2$ points and beyond, it is not always possible to find a sphere that will shatter these points, so the VC dimension must cap out at $n + 1$.
2. The VC Dimension of this is 5. 
   ![[Pasted image 20231017235205.png]]
   ![[Pasted image 20231019181904.png]]

To find the minimum sample size required, we can use the formula:
$m \geq \frac{1}{\varepsilon}(4\ln(\frac{2}{\delta}) + 8 * \text{VC(H)} \ln(\frac{13}{\epsilon})$

After plugging in the values of $\delta = 0.05, \epsilon = 0.2, \text{VC(H) = 5}$, we get:
$m \geq \frac{1}{0.2}(4\ln(\frac{2}{0.05}) + (8 * 5) \ln(\frac{13}{0.2})$

Simplifying, we get
$m \geq 908.655$

So $m = 909$
# Problem 2
1. 
   ![[Pasted image 20231019223949.png]]
   ![[Pasted image 20231019225441.png]]
   ![[Pasted image 20231019225636.png]]
   ![[Pasted image 20231019225707.png]]
   ![[Pasted image 20231019230041.png]]
   ![[Pasted image 20231019230115.png]]
   ![[Pasted image 20231019230147.png]]
   ![[Pasted image 20231019230228.png]]
   ![[Pasted image 20231019230309.png]]
   ![[Pasted image 20231019230336.png]]
2. 
   ![[Pasted image 20231018111556.png]]
   ![[Pasted image 20231018111654.png]]
3. 
   My optimal values for alpha were:
   
   [-7.57, -0.013, 2.21 , 0, 0.49, 0. , 5.28, 0. , -0.81, 0. , -0.32, 0 , -0.13, 0, -0.6 , 0, -4.89, 0, 3.59, 0, -2.08, 0, -0.39, 0. , 0.45, 0. , -0.79, 0. , -3.06 , 0. , -1.19, 0. , -0.74, 0. , -2.95, 0. , -1.90, 0. , -0.05 , 0. , -0.32, 0. , -0.16, 0. , -0.38, 0. ]
   
   My value of the exponential loss was 39.51
4. The accuracy of bagging is lower with Training Accuracy being 72.5% and Test Accuracy being 61.5%. This is much lower accuracy than coordinate descent and adaBoost. 
5. For this dataset, adaBoost is preferred since it yields similar accuracy to coordinate descent, but requires less iterations for the similar accuracy. 


