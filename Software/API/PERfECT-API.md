# PERfECT API

Xinyu Tian,  Bakitchi

[中文](README_CN.md)

> This is a API introduction for PERfECT system.

## Introduction

[PERfECT](https://github.com/WISE-PERfECT/PERfECT) The world's first and best ECT control system.

Electrical Chemical Transistor (ECT) is a special type of transistor that has a totally different work mechanism compared with traditional Si-based transistors. Essentially, this kind of transistor is playing more with ions but not electrons, which makes it more similar to a human neural (both sensing and computing) system.

PERfECT start from a project called 'Personalized Electrical Reader for Electrical Chemical Transistors' which aims to prototype the world's first characterization system specifically for ECT devices, with our 10 years of dedication to it.

## Install

1. [Buy](xxx) one or more PERfECT the GUI from xxx.
1. [Download](https://github.com/WISE-PERfECT/PERfECT) the GUI from Github.
2. Say `"PERfECT yyds"`, `"HKU WISE yyds"` loudly 10 times.
4. Then follow this instruction and it should be work now.

## See what's PERfECT

![PERfECT F1](https://github.com/WISE-PERfECT/PERfECT/blob/main/figures/20210822213950.jpg?raw=true)



> **Thanks**: This `vue-dark.css` by [typora-vue-dark-theme](https://github.com/MamoruDS/typora-vue-dark-theme).

## Electrochemistry

### SWV

<hr/>

`SWV` is a xxx

#### [#]() swvconfig  

- **Usage:**
```haskell
swvconfig InitEmV, FinalEmV, IncrEmV, AmplitudemV, FrequencyHz, QuietTimeS, SensitivityNA; 
```
- **Default:**
```haskell
swvconfig -500,0,4,25,200,2,1000; 
```
Furthers Ex Furthers Ex Furthers Ex Furthers Ex Furthers Ex Furthers Ex Furthers Ex Furthers Ex 

- **Return:** `Status Code; Message`

```haskell
E.g.:

66;Succssful config

100;Illegal arguments

200;Unsupported commands
```

- **Arguments:**
> ![Screenshot 5](https://github.com/MamoruDS/typora-vue-theme/raw/master/screenshots/screenshot_02.png)
>


#### swvmeas;

![Screenshot 4](https://github.com/MamoruDS/typora-vue-theme/raw/master/screenshots/screenshot_01.png)

![Screenshot 5](https://github.com/MamoruDS/typora-vue-theme/raw/master/screenshots/screenshot_02.png)

## Transistors

### Transfer curve

<hr/>

`Transfer curve` is a xxx

#### [#]() transconfig 

- **Usage:**

```haskell
transconfig InitVgsmV,FinalVgsmV,IncrVgsmV,IncrTimemS,VdsmV,SensitivityLevel,Hysteresis;
```

- **Default:**

```haskell
transconfig -200,800,5,1000,-600,1,1;
```

Furthers Ex Furthers Ex Furthers Ex Furthers Ex Furthers Ex Furthers Ex Furthers Ex Furthers Ex 

- **Return:** `Status Code; Message`

```haskell
E.g.:

66;Succssful config

100;Illegal arguments

200;Unsupported commands
```

- **Arguments:**

> ![Screenshot 5](https://github.com/MamoruDS/typora-vue-theme/raw/master/screenshots/screenshot_02.png)

### Transient responds

<hr/>

`Transient responds` is a xxx

#### [#]() transientcfg

- **Usage:**

```haskell
transientcfg InitVgsmV,InitVdsmV,InitQuietTimemS,BitNumber,BitPattern,VgsPulseHmV,VgsPulseLmV,PulseWidthmS,FinalVgsmV,FinalQuietTimemS,SensitivityLevel;
```

- **Default:**

```haskell
transientcfg 0,-600,5000,1,1,800,0,50,-100,10000,1;
transientcfg 0,-600,5000,4,12,800,0,50,-100,10000,1;
```

Furthers Ex Furthers Ex Furthers Ex Furthers Ex Furthers Ex Furthers Ex Furthers Ex Furthers Ex 

![PERfECT_transient F5](https://github.com/WISE-PERfECT/PERfECT/blob/main/figures/transient.png?raw=true)

- **Return:** `Status Code; Message`

```haskell
E.g.:

66;Succssful config

100;Illegal arguments

200;Unsupported commands
```

- **Arguments:**

> 

