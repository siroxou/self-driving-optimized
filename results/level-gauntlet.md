Reference driver (reads the simulator, ignores sensors):

| Sensors | Success | Crash | Off road | Pedestrian hit | Pothole damage | Out of time | Route completion | Potholes per run |
|---|---|---|---|---|---|---|---|---|
| reference driver | 100% | 0% | 0% | 0% | 0% | 0% | 100% | 3.0 |

Viewer's Auto driver (best model for each sensor set):

| Sensors | Success | Crash | Off road | Pedestrian hit | Pothole damage | Out of time | Route completion | Potholes per run |
|---|---|---|---|---|---|---|---|---|
| `odom,lane,nav,lidar` | 0% | 32% | 0% | 0% | 68% | 0% | 21% | 0.7 |
| `odom,lidar,nav` | 0% | 28% | 0% | 6% | 66% | 0% | 21% | 0.7 |
| `odom,lidar,gps` | 0% | 38% | 0% | 2% | 60% | 0% | 18% | 0.6 |
| `odom,lane,nav,lidar,sonar,radar` | 0% | 28% | 0% | 4% | 68% | 0% | 22% | 0.7 |
| `odom,lane,nav,radar` | 0% | 36% | 0% | 10% | 54% | 0% | 15% | 0.6 |
| `odom,sonar,radar,nav` | 0% | 32% | 0% | 12% | 56% | 0% | 17% | 0.6 |
| `lidar` | 0% | 26% | 2% | 6% | 66% | 0% | 21% | 0.7 |
| `odom,lane,nav` | 0% | 40% | 0% | 2% | 58% | 0% | 15% | 0.6 |
