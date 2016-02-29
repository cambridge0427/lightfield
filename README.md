## Continuous Depth Map Reconstruction From Light Fields

This project is for the research experiments for this paper [[pdf]](https://www.cs.sfu.ca/~li/papers-on-line/Light-fields-TIP2015.pdf)
>Continuous depth map reconstruction from light fields  
>J. Li, M. Lu, and Z.N. Li,  
>IEEE Trans. on Image Processing, 24(11), 2015, 3257-3265.  

The code base contains two executable parts. Part I is for initial depth estimation, which is described in section II and III in the paper. The part II is to optimize the initial depth map described in section IV in the paper.

### Part I: Initial estimation

**How to compile**:

+ Required libraries: OpenCV2 HDF VIGRA  
    It might have some compatibility issues with OpenCV3. Sorry...  
    If you are on mac:  
    ```
    brew install homebrew/science/opencv
    brew install homebrew/science/vigra
    ```
 
+ Compile:
    ```
    cd <root directory of this project>/initialEstimation
    OPENCV=<your opencv library directory> HDF=<your hdf library directory> VIGRA=<your vigra library directory> make
    ```

**How to use**:

+ Input data:
    * Download data from the light field dataset provided by

        >Datasets and Benchmarks for Densely Sampled 4D Light Fields  
        >S. Wanner, S. Meister, B. Goldluecke  
        >In Vision, Modelling and Visualization (VMV), 2013.

        At the writing time, they are updating the site and the data might not be available  
        [http://www.informatik.uni-konstanz.de/cvia/resources/](http://www.informatik.uni-konstanz.de/cvia/resources/).  
        Old URL:  
        [http://hci.iwr.uni-heidelberg.de/HCI/Research/LightField/lf_benchmark.php](http://hci.iwr.uni-heidelberg.de/HCI/Research/LightField/lf_benchmark.php)  

    * Put the h5 files under ./data/input/

+ Run:
    ```
    cd <root directory of this project>/initialEstimation
    ./initial_estimation <data name>
    ```
+ Output:  
    Find the output data under ./data/initial_results/

### Part II: Optimization

**How to compile**:

+ Required libraries: OpenCV2 HDF Eigen  

+ Compile:
    ```
    cd <root directory of this project>/optimization
    OPENCV=<your opencv library directory> HDF=<your hdf library directory> EIGEN=<your eigen library directory> make
    ```

**How to use**:

+ Input data:  
    Make sure the output data from the first part is ready under ./data/initial_results/

+ Run:
    ```
    cd <root directory of this project>/optimization
    ./optimize <data name> <use revised initial results> <optimization method> [<segment>]
    ```
    * **data name**: without extension. eg, buddha.  
    * **use revised initial results**: 0 or 1  
        Use revised initial result or original result from structure tensor  
    * **optimization method**:  
        NoOpt = 0: no optimization  
        MRFOpt = 1: use Markov Random Field  
        FastMatting = 2: use fasting matting to solve the linear equation  
        ClassicOpt = 3: use Conjugate Gradient to solve the linear equation  
        MixOpt = 4: use mixed method to solve the linear equation with segmentation
    * **segment**: 0 or 1  
        This is only applicable for FastMatting, ClassicOpt and MixOpt  

+ Output:  
    Find the output data under ./data/output/

### Citation

+ The project runs on the dataset provided by
>Datasets and Benchmarks for Densely Sampled 4D Light Fields  
>S. Wanner, S. Meister, B. Goldluecke  
>In Vision, Modelling and Visualization (VMV), 2013.

+ Mean Shift Analysis Library is included, which is based on these papers
>D. Comaniciu, P. Meer: Mean Shift: A robust approach toward feature space analysis.  
>C. Christoudias, B. Georgescu, P. Meer: Synergism in low level vision.

+ MRF energy minimization software is included, which was published accompanying this paper
>A Comparative Study of Energy Minimization Methods for Markov Random Fields.  
>R. Szeliski, R. Zabih, D. Scharstein, O. Veksler, V. Kolmogorov, A. Agarwala, M. Tappen, and C. Rother.  
>In Ninth European Conference on Computer Vision (ECCV 2006), volume 2, pages 16-29, Graz, Austria, May 2006.  

