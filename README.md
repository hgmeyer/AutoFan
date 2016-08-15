# AutoFan
![](https://cdn.hackaday.io/images/874361467020583130.png)

AutoFan is a prototype for controlling the direction of air flow of a fan based on computer vision in order to avoid fatigue during long distance car drives. It uses a *low-cost camera* (e.g. a webcam), a *Raspberry Pi 2*, a *face recognition algorithm*, an algorithm for *eye blink detection* and two *servo motors* controlling the lamellae of a *custom-made fan*. By inferring the position of a face from the camera images the servo motor angles are adjusted to point the airflow into (or away) from the face. By measuring eye blink frequency the system automatically points the air flow into the direction of the driver's face if the driver has been detected to be tired.

**Please note:** This is the repository for a project hosted on [hackaday.io](http://hackaday.io/). Please visit the [project page](https://hackaday.io/project/12384-autofan-automated-air-flow-direction-control) for more information.

<a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-sa/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/">Creative Commons Attribution-ShareAlike 4.0 International License</a>.
