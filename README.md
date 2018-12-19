# Minimum-Cut Clustering
a python tool to analyze exciton networks in photosynthetic systems

## Introduction
- The minimum-cut clustering method is developed by Wei-Hsiang Tseng (Dimsplendid) in his master thesis 
  "Theoretical Analysis of Energy Transfer Networks in Photosynthetic Systems"
  (https://hdl.handle.net/11296/47fh6w) 
  @ Yuan-Chung Cheng's group (http://quantum.ch.ntu.edu.tw/ycclab/)
##### See also
 simpleNA (https://github.com/dimsplendid/simpleNA), the predecessor, in C

##### Reference
- De-Wei Ye, Wei-Hsiang Tseng and Yuan-Chung Cheng. 
  Systematic coarse-graining of photosynthetic energy transfer networks. 
  *in preparation.*

## Requirements
- This tool is wrote and test on **Python 3.7**
- For python modules: see `requirements.txt`
- **Incompatible** with Python 2.X
- To plot graph, **graphviz** is needed, download: https://www.graphviz.org

## Usage

### Command Line

    python main.py [file] [job name] -[options] [dictionary]=[value] --[keywords]

use `python main.py -h` or `cat doc/usage` for more details

### Input File

#### Hamiltonian

If the input file name ends with '.H', it will be regarded as Hamiltonian file

    (Site Name) (n sites, optional)
    Hamintonian (n * n matrix)

##### Note:
- Hamiltonian must be symmetric
- Parameters for rate constant calculation is required when this program is running.

#### Rate Constants Matrix

    (State Name) (n, optional)
    State Energy (n)
    Rate Constant Matrix (n * n matrix)
    
e.g. 3 states input:

    A B C
    100 200 300
    -1   2   3
     1  -2   1
     0   0  -4
     
##### Note:
- Diagonal terms and negative terms are ignored.
- The j, i term represents rate from i to j. This logic also applies on flow matrix output. 
    
#### Site Position
    (Site Name) x, y, z (n sites, site name is optional)
    
e.g. 3 sites input:

    site1     0    0    0
    site2   9.2  426  689
    site3   7.7  847  609   
    
Note that the site names should match the provided names in the Hamiltonian input file.
