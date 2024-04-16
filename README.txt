Shoal of fishes by Piotr Gugnowski 320690

-------------------------------------------------------------------------------------------
IMPORTANT
-------------------------------------------------------------------------------------------
Run program in RELEASE mode!
If you are lacking some libraries/include files go to properies of a project,
in VC++ Directories add 'Incude Directories' folder that I put inside project
(fish-shoals-gpu\fish_shoals-gpu\includes).

Respectfully add proper folder to 'Library directories' 
(fish-shoals-gpu\fish_shoals-gpu\libraries).
-------------------------------------------------------------------------------------------


-------------------------------------------------------------------------------------------
CONFIGURATION
-------------------------------------------------------------------------------------------
GPU - by default
CPU - go to config.cuh file and uncomment #define CPU at line 5

number of fishes - find and change lines 49 and 47 in config.cuh (#define FISH_COUNT 50000)

other meaningful parameters can be changes in program's gui
-------------------------------------------------------------------------------------------


-------------------------------------------------------------------------------------------
OPTIMALIZATION
-------------------------------------------------------------------------------------------
Mainly I use dynamically adjustable to fishes view range unit grid, so every fish only 
check its close neighbourhood, not all the fishes in the simulation.
-------------------------------------------------------------------------------------------


-------------------------------------------------------------------------------------------
VIDEO
-------------------------------------------------------------------------------------------
[![Video Thumbnail](https://i.ytimg.com/vi/P9PdMS1-Ul0/maxresdefault.jpg)](https://youtube.com/shorts/P9PdMS1-Ul0?feature=shared)
-------------------------------------------------------------------------------------------

