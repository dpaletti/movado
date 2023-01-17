
# Table of Contents

1.  [Installation](#org32f73e8)
2.  [Usage](#org407d412)
3.  [Parameters](#org6d82707)



<a id="org32f73e8"></a>

# Installation

Movado needs python 3.6 or higher. Install it through pip (on many linux systems you need to use pip3 to force python3 installation):

    pip3 install --user --no-cache movado


<a id="org407d412"></a>

# Usage

Movado exposes just an annotation which you need to apply to the fitness function you want to approximate:

    @approximate(outputs=3)
    def fitness(point: List[Number]) -> List[Number]:
        # my fitness logic
    
    class SomeClass:
        def __init__():
            ...
        @approximate(outputs=1)
        def fitness(self, point: List[Number]) -> List[Number]:
            # this is also allowed

The only mandatory parameter to approximate is the **number of objectives** of your optimization problem, in the above case the fitness function returns a list of 3 floats and takes as input a list of numbers.
**Fitness Function Structure**:

1.  the fitness function must take only a list of numbers as input. If we approximate a method in a class `self` is allowed as first parameter but it must be strictly followed by the list of numbers
2.  the fitness function must output a list of numbers with constant length between fitness evaluation


<a id="org6d82707"></a>

# Parameters

Other than `outputs` several other optional parameters are available

<table border="2" cellspacing="0" cellpadding="6" rules="groups" frame="hsides">


<colgroup>
<col  class="org-left" />

<col  class="org-left" />

<col  class="org-left" />
</colgroup>
<thead>
<tr>
<th scope="col" class="org-left">Parameter</th>
<th scope="col" class="org-left">Description</th>
<th scope="col" class="org-left">Default Value</th>
</tr>
</thead>

<tbody>
<tr>
<td class="org-left">disabled</td>
<td class="org-left">if True the approximation</td>
<td class="org-left">False</td>
</tr>


<tr>
<td class="org-left">&#xa0;</td>
<td class="org-left">is disabled</td>
<td class="org-left">&#xa0;</td>
</tr>
</tbody>

<tbody>
<tr>
<td class="org-left">stochastic</td>
<td class="org-left">if True caching of fitness</td>
<td class="org-left">False</td>
</tr>


<tr>
<td class="org-left">&#xa0;</td>
<td class="org-left">calls is disabled</td>
<td class="org-left">&#xa0;</td>
</tr>
</tbody>

<tbody>
<tr>
<td class="org-left">estimator</td>
<td class="org-left">estimator to use</td>
<td class="org-left">HoeffdingAdaptiveTree</td>
</tr>
</tbody>

<tbody>
<tr>
<td class="org-left">controller</td>
<td class="org-left">controller to user</td>
<td class="org-left">Mab</td>
</tr>
</tbody>
</table>

# Associated Publication

```
@inproceedings{paletti2021online,
  title={Online Learning RTL Synthesis for Automated Design Space Exploration},
  author={Paletti, Daniele and Peverelli, Francesco and Conficconi, Davide and Santambrogio, Marco D},
  booktitle={2022 IEEE International Parallel and Distributed Processing Symposium Workshops (IPDPSW)},
  year={2022},
  organization={IEEE}
}
```

