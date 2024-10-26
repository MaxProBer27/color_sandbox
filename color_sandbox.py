import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import sympy as sp
from scipy import signal

#np.random.seed(1)

class color():
    def __init__(self,r,g,b) -> None:
        self.values = np.array([r,g,b])
    
    def norm(self):
        return np.sqrt(((self.values)**2).sum())
    
    def norm1(self):
        return abs(self.values.sum())
    
    def norm_inf(self):
        return max(self.values)

    def distance(self, c2):
        return color(*((self.values - c2.values)**2)).norm()
    
    def visual_distance(self,c2):
        matrix = np.array([[self.values for i in range(10)] +
                           [c2.values for i in range(10)]]*10)
        plt.axis('off')
        plt.imshow(matrix)
        plt.show()
        print(self.distance(c2))
    
    def intensity_distance(self, c2):
        return self.black_white().distance(c2.black_white())
    
    def black_white(self,dist = 'euclid'):
        norm = ['euclid','one','inf'].index(dist)
        norm_method = [self.norm, self.norm1, self.norm_inf][norm]
        normal_constant = [color(1,1,1).norm(),
                           color(1,1,1).norm1(),
                           color(1,1,1).norm_inf()][norm]
        return color(*np.array([norm_method()/normal_constant]*3))


class image():
    """Class of images
    """    
    def __init__(self,img) -> None:
        if type(img) == str:
            if mpimg.imread(img).dtype == 'uint8':
                self.matrix = mpimg.imread(img)/255
            else:
                self.matrix = mpimg.imread(img)
        elif type(img) == np.ndarray:
            self.matrix = img
        elif type(img) == list:
            self.matrix = np.array(img)
    
    def __str__(self):
        return str(self.matrix)
    
    def __repr__(self):
        return str(self)
    
    def rgb(self):
        R = np.array([[pixel[0] for pixel in row] for row in self.matrix])
        G = np.array([[pixel[1] for pixel in row] for row in self.matrix])
        B = np.array([[pixel[2] for pixel in row] for row in self.matrix])
        return (R,G,B)

    def save(self,name = 'output.png'):
        plt.axis('off')
        plt.imsave(name,self.matrix)

    def show(self):
        plt.axis('off')
        plt.imshow(self.matrix)
        plt.show()
    
    def __add__(self,i2):
        suma = image(self.matrix + i2.matrix)
        RGB = suma.rgb()
        for matrix in RGB: # increase contrast for all borders and normalize all values
                matrix[matrix < 0] = 0.
                matrix[matrix > 1] = 1.
        suma = [[[RGB[0][i,j],RGB[1][i,j],RGB[2][i,j]]
                  for j in range(self.matrix.shape[1])]
                  for i in range(self.matrix.shape[0])]
        return image(suma)
    
    def __sub__(self,i2):
        resta = image(self.matrix - i2.matrix)
        RGB = resta.rgb()
        for matrix in RGB: # increase contrast for all borders and normalize all values
                matrix[matrix < 0] = 0.
                matrix[matrix > 1] = 1.
        resta = [[[RGB[0][i,j],RGB[1][i,j],RGB[2][i,j]]
                  for j in range(self.matrix.shape[1])]
                  for i in range(self.matrix.shape[0])]
        return image(resta)
    
    def Gaussian_blur(self, sigma = 5):
        """Returns an object of the class 'image' with the Gaussian filter
        applied to the original image.

        Args:
            sigma (number): Standard devation

        Returns:
            image: Image with the Gaussian filter applied
        """        
        x,y = sp.symbols('x,y')
        gauss = (np.e**(-(x**2 + y**2)/(2*sigma**2)))/(2*np.pi*sigma**2)
        G = sp.lambdify([x,y],gauss)
        n = int((np.ceil(6*sigma)-1)/2) # Matrix typically is of dimensions 6sigma x 6sigma
        kernel = np.array([[G(i,j) for j in range(-n,n+1)] for i in range(-n,n+1)])
        kernel = kernel/kernel.sum() # Normalization of kernel
        rgb_blur = [signal.convolve2d(matrix, kernel, mode = 'same')
                    for matrix in self.rgb()]
        blur = [[[rgb_blur[0][i,j],rgb_blur[1][i,j],rgb_blur[2][i,j]] 
                for j in range(self.matrix.shape[1])] 
                for i in range(self.matrix.shape[0])]
        return image(blur)

    def Laplacian(self, full = True, positive = False, Gauss = 0):
        """Returns an object of the class 'image' with the Laplacian filter
        applied to the original image.

        Args:
            full (bool, optional): If 'full' is True, it uses all neighboors to
            approximate the second derivative; else, it uses only adjacent
            neighboors. Defaults to False.

            positive (bool, optional): If 'positive' is True, multpiles kernel times -1.
            Defaults to False.

            Gauss (int, optional): Value of 'sigma' if Gaussian blur is needed.
            Defaults to 0.

        Returns:
            image: Image with Laplacian filter applied
        """        
        if full:
            kernel = np.array([[-1,-1,-1],
                            [-1,8,-1],
                            [-1,-1,-1]])*(-1)**positive
        else:
            kernel = np.array([[0,-1,0],
                            [-1,4,-1],
                            [0,-1,0]])*(-1)**positive
        if Gauss:
            rgb_blur = self.Gaussian_blur(sigma=Gauss).rgb()
            rgb_laplace = [signal.convolve2d(matrix, kernel, mode = 'same')*Gauss**2
                           for matrix in rgb_blur]
            for matrix in rgb_laplace: # increase contrast for all borders and normalize all values
                matrix[matrix < 0] = 0.
                matrix[matrix > 1] = 1.
            laplace = [[[rgb_laplace[0][i,j],rgb_laplace[1][i,j],rgb_laplace[2][i,j]]
                         for j in range(self.matrix.shape[1])]
                         for i in range(self.matrix.shape[0])]
        else:
            rgb_laplace = [signal.convolve2d(matrix, kernel, mode = 'same')
                           for matrix in self.rgb()]
            for matrix in rgb_laplace: # increase contrast for all borders and normalize all values
                matrix[matrix < 0] = 0.
                matrix[matrix > 1] = 1.
            laplace = [[[rgb_laplace[0][i,j],rgb_laplace[1][i,j],rgb_laplace[2][i,j]]
                         for j in range(self.matrix.shape[1])]
                         for i in range(self.matrix.shape[0])]
        return image(laplace)
    
    def Black_White(self, dist = 'euclid'):
        """Returns the image transformed to black and white

        Args:
            dist (str, optional): Norm function to be used. Defaults to 'euclid';
            can be 'inf', 'one' or 'euclid'

        Returns:
            image: The image transformed to black and white
        """        
        if dist == 'inf':
            bw = [[[self.matrix[i,j].max()]*3 for j in range(self.matrix.shape[1])] 
                for i in range(self.matrix.shape[0])]
        elif dist == 'euclid':
            bw = [[[color(*self.matrix[i,j]).norm()/np.sqrt(3)]*3
                    for j in range(self.matrix.shape[1])] 
                    for i in range(self.matrix.shape[0])] # Divide by sqrt(3) to normalize
        elif dist == 'one':
            bw = [[[color(*self.matrix[i,j]).norm1()/3]*3
                    for j in range(self.matrix.shape[1])] 
                    for i in range(self.matrix.shape[0])] # Divide by 3 to normalize
        return image(bw)

    def brightness_reduction(self,r):       
        br = [[self.matrix[i,j]/r
                for j in range(self.matrix.shape[1])] 
                for i in range(self.matrix.shape[0])]
        return image(br)
    
    def pallete(self,n = 6):
        """Returns an image's pallete of colors and an image of the pallete

        Args:
            n (int, optional): Number of colors wanted. Defaults to 6.

        Returns:
            tuple: first entry is the image of the pallete, second entry is a numpy array
            contaning the pallete's colors.
        """  
        pixels = [(np.random.randint(self.matrix.shape[0]), 
                   np.random.randint(self.matrix.shape[1]))
                   for i in range(1000)]
        colors = [color(*self.matrix[pixel]) for pixel in pixels]
        pallete_colors = []
        for c in colors:
            if c not in pallete_colors and all(c.distance(x) > 0.01 for x in pallete_colors):
                pallete_colors.append(c)
        if len(pallete_colors) < n:
            pallete_colors = np.array([[pallete_colors[i].values
                                        for i in range(len(pallete_colors))]]*n)
        else:
            pallete_colors = np.array([[pallete_colors[i].values for i in range(n)]]*n)
        return (image(pallete_colors),pallete_colors)
    
    def color_extraction(self, c):
        pass
    
    def pallete_transfer(self, root_image,n = 6,diag = False):
        transfer = root_image.pallete(n)
        transfer[0].show() 
        pallete = transfer[1]
        colors = [color(*c) for c in pallete[0]]
        copy = np.copy(self.matrix)
        ratio = self.matrix.shape[0]/self.matrix.shape[1]
        for row in range(self.matrix.shape[0]):
            interval = int(np.floor(row/(ratio) + 0.3)) if diag else self.matrix.shape[1] # Add 0.3 to ratio to avoid dimension error
            for col in range(interval):
                pixel = color(*self.matrix[row,col])
                colors.sort(key=lambda c:c.distance(pixel))
                new_color = colors[0]
                copy[row,col] = new_color.values
        return image(copy)
    
    def invert(self):
        RGB = self.rgb()
        inverse = [[[1-RGB[0][i,j],1-RGB[1][i,j],1-RGB[2][i,j]]
                     for j in range(self.matrix.shape[1])]
                     for i in range(self.matrix.shape[0])]
        return image(inverse)

    def cartoon_filter(self):
        pass

I1 = image('/home/maximiliano/Downloads/Pictures/japgruv1.png')
I2 = image('/home/maximiliano/Downloads/Pictures/japgruv2.png')

I1.show()
I2.show()

I1.brightness_reduction(2).show()
I2.brightness_reduction(2).show()