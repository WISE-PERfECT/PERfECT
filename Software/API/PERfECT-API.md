# PERfECT API

Xinyu Tian

[中文](README_CN.md)

> This is a API introduction for PERfECT system.

## Introduction

[PERfECT](https://github.com/WISE-PERfECT/PERfECT) The world's first and best ECT control system.

Electrical Chemical Transistor (ECT) is a special type of transistor that has a totally different work mechanism compared with traditional Si-based transistors. Essentially, this kind of transistor is playing more with ions but not electrons, which makes it more similar to a human neural (both sensing and computing) system.

PERfECT start from a project called 'Personalized Electrical Reader for Electrical Chemical Transistors' which aims to prototype the world's first characterization system specifically for ECT devices, with our 10 years of dedication to it.

![PERfECT F0](https://github.com/WISE-PERfECT/PERfECT/blob/main/figures/PERfECT-FUNCS.png?raw=true)

## Install

1. [Buy](xxx) one or more PERfECT from xxx.
1. [Download](https://github.com/WISE-PERfECT/PERfECT) the GUI from Github.
2. Say `"PERfECT yyds"`, `"HKU WISE yyds"` loudly 10 times.
4. Then follow this instruction and it should be work now.

## See what's PERfECT

![PERfECT F1](https://github.com/WISE-PERfECT/PERfECT/blob/main/figures/20210822213950.jpg?raw=true)



> **Thanks**:  [Bakitchi](bakitchi@connect.hku.hk) for all your advice :beer:. :beer:. :cat: :girl:.



## Basial Command

### Stop Everything

<hr/>

`stop`stop is a xxx

#### [#]() stop  

- **Usage:**

```haskell
stop 1;
```

- **Default:**

```haskell
stop 1;
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

> 

####



### Get temperature

<hr/>

`gettemp` is a xxx

#### [#]() stop  

- **Usage:**

```haskell
gettemp 1;
```

- **Default:**

```haskell
gettemp 1;
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

> 

####  

## Electrochemistry

### Chronoamperometry

<hr/>

`I-T curve` is a xxx

#### [#]() ecitcfg  

- **Usage:**

```haskell
ecitcfg  InitEmV, SamplingRatemS,SamplingTimeS,SensitivityLevel;
```

- **Default:**

```haskell
ecitcfg -200,100,0,1;
```

| Parameter        | Explain                                          | Unit | Range (ALL int) |
| ---------------- | ------------------------------------------------ | ---- | --------------- |
| InitEmV          | The  initial voltage applied to Gate and Source. | mV   | -1000 to +1000  |
| SamplingRatemS   | The sampling rate of current (Is) monitoring.    | mS   | 10-60000        |
| SamplingTimeS    | The sampling time, can be infinite when = 0      | S    | 0 or 1-600      |
| SensitivityLevel | Sensitivity                                      | N/A  | 0 to 10         |

Furthers Ex Furthers Ex Furthers Ex Furthers Ex Furthers Ex Furthers Ex Furthers Ex Furthers Ex 

- **Return:** `Status Code; Message`

```haskell
E.g.:

66;Succssful config

100;Illegal arguments

200;Unsupported commands
```

- **Arguments:**

> 

### SWV

<hr/>

`SWV` is a xxx

#### [#]() swvconfig  

- **Usage:**
```haskell
swvconfig InitEmV, FinalEmV, IncrEmV, AmplitudemV, FrequencyHz, QuietTimeS, Sweepback,SensitivityLevel; 
```
- **Default:**
```haskell
swvconfig -500,0,4,25,200,2,0,1; 
```

| Parameter        | Explain                                                      | Unit | Range (ALL int) |
| ---------------- | ------------------------------------------------------------ | ---- | --------------- |
| InitEmV          | The  initial voltage applied to Gate and Source.             | mV   | -1000 to +1000  |
| FinalEmV         | The  initial voltage applied to Drain and Source.            | mV   | -1000 to +1000  |
| IncrEmV          | Time before Pulse start.                                     | mS   | 10-60000        |
| FrequencyHz      | The sampling time, can be infinite when = 0                  | N/A  | 0-60000         |
| QuietTimeS       | Pulse train pattern. E.g. when bit number = 4, BitPatterm = 10, then pulse train will be 0110. | N/A  | 0-60000         |
| Sweepback        |                                                              | N/A  | 0/1             |
| SensitivityLevel | Sensitivity                                                  | N/A  | 0 to 7          |


- **Return:** `Status Code; Message`

```haskell
E.g.:

66;Succssful config

100;Illegal arguments

200;Unsupported commands
```

- **Arguments:**
> 
>



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


| Parameter        | Explain                                          | Unit | Range (ALL int) |
| ---------------- | ------------------------------------------------ | ---- | --------------- |
| InitVgsmV        | The  initial voltage applied to Gate and Source. | mV   | -1000 to +1000  |
| FinalVgsmV       | The final voltage applied to Gate and Source.    | mV   | -1000 to +1000  |
| IncrVgsmV        |                                                  | mV   |                 |
| IncrTimemS       |                                                  | mS   |                 |
| VdsmV            | The voltage applied to Drain and Source.         | mV   | -1000 to +1000  |
| SensitivityLevel | Sensitivity                                      | N/A  | 0 to 10         |
| Hysteresis       |                                                  | N/A  | 0/1             |


- **Return:** `Status Code; Message`

```haskell
E.g.:

66;Succssful config

100;Illegal arguments

200;Unsupported commands
```

- **Arguments:**

> 

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

| Parameter        | Explain                                                      | Unit | Range (ALL int) |
| ---------------- | ------------------------------------------------------------ | ---- | --------------- |
| InitVgsmV        | The  initial voltage applied to Gate and Source.             | mV   | -1000 to +1000  |
| InitVdsmV        | The  initial voltage applied to Drain and Source.            | mV   | -1000 to +1000  |
| InitQuietTimemS  | Time before Pulse start.                                     | mS   | 10-60000        |
| BitNumber        | The sampling time, can be infinite when = 0                  | N/A  | 0-60000         |
| BitPattern       | Pulse train pattern. E.g. when bit number = 4, BitPatterm = 10, then pulse train will be 0110. | N/A  | 0-60000         |
| VgsPulseHmV      | Pulse voltage when bit in pulse train  = 1.                  | mV   | -1000 to +1000  |
| VgsPulseLmV      | Pulse voltage when bit in pulse train  = 0.                  | mV   | -1000 to +1000  |
| PulseWidthmS     | Pulse width.                                                 | mS   | 1-60000         |
| FinalVgsmV       | The final voltage (after last pulse) applied to Gate and Source. | mV   | -1000 to +1000  |
| FinalQuietTimemS | Time before Pulse start.                                     | mS   | 10-60000        |
| SensitivityLevel | Sensitivity                                                  | N/A  | 0 to 10         |



Furthers Ex Furthers Ex Furthers Ex Furthers Ex Furthers Ex Furthers Ex Furthers Ex Furthers Ex 

![PERfECT_transient F5](https://github.com/WISE-PERfECT/PERfECT/blob/main/figures/transient.png?raw=true)


> **Thanks**:  [Soft Rocks](dingyao6@connect.hku.hk) for awsome figures :beer:.

- **Return:** `Status Code; Message`

```haskell
E.g.:

66;Succssful config

100;Illegal arguments

200;Unsupported commands
```

- **Arguments:**

### Chronoamperometry

<hr/>

`Chronoamperometry` means xxxx

#### [#]() caectcfg

- **Usage:**

```haskell
caectcfg InitVgsmV,InitVdsmV,SamplingRatemS,SamplingTimeS,SensitivityLevel;
```

- **Default:**

```haskell
caectcfg 200,-300,60,5,1;
```

| Parameter        | Explain                                           | Unit | Range (ALL int) |
| ---------------- | ------------------------------------------------- | ---- | --------------- |
| InitVgsmV        | The  initial voltage applied to Gate and Source.  | mV   | -1000 to +1000  |
| InitVdsmV        | The  initial voltage applied to Drain and Source. | mV   | -1000 to +1000  |
| SamplingRatemS   | The sampling rate of current (Is) monitoring.     | mS   | 10-60000        |
| SamplingTimeS    | The sampling time, can be infinite when = 0       | S    | 0 or 1-600      |
| SensitivityLevel | Sensitivity                                       | N/A  | 0 to 10         |



- **Return:** `Status Code; Message`

```haskell
E.g.:

66;Succssful config

100;Illegal arguments

200;Unsupported commands
```

- **Arguments:**

> 

### "Wash" the transistor

<hr/>

`Wash transistor` means xxxx

#### [#]() washectcfg

- **Usage:**

```haskell
washectcfg InitVgsmV,InitVdsmV,VgsPulseHmV,VgsPulseLmV,PulseVgsWidth,WashTimemS,SensitivityLevel;
```

- **Default:**

```haskell
washectcfg 0,-300,200,-200,5,50000,1;
```

InitVdsmV:

PulseVgsHmV

PulseVgsLmV

PulseVgsWidth

WashTimemS

SensitivityLevel;



 Ex Furthers Ex Furthers Ex Furthers Ex Furthers Ex Furthers Ex Furthers Ex Furthers Ex 

- **Return:** `Status Code; Message`

```haskell
E.g.:

66;Succssful config

100;Illegal arguments

200;Unsupported commands
```

- **Arguments:**

> 