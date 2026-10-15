Reference driver (reads the simulator, ignores sensors):

| Sensors | Success | Crash | Off road | Pedestrian hit | Pothole damage | Out of time | Route completion | Potholes per run |
|---|---|---|---|---|---|---|---|---|
| reference driver | 100% | 0% | 0% | 0% | 0% | 0% | 100% | 0.8 |

Viewer's Auto driver (best model for each sensor set):

| Sensors | Success | Crash | Off road | Pedestrian hit | Pothole damage | Out of time | Route completion | Potholes per run |
|---|---|---|---|---|---|---|---|---|
| `odom,lane,nav,lidar` | 6% | 12% | 0% | 16% | 66% | 0% | 41% | 0.7 |
| `odom,lidar,nav` | 4% | 24% | 0% | 10% | 62% | 0% | 37% | 0.6 |
| `odom,lidar,gps` | 2% | 30% | 0% | 12% | 56% | 0% | 34% | 0.6 |
| `odom,lane,nav,lidar,sonar,radar` | 6% | 2% | 0% | 16% | 76% | 0% | 45% | 0.8 |
| `odom,lane,nav,radar` | 0% | 66% | 0% | 2% | 32% | 0% | 35% | 0.4 |
| `odom,sonar,radar,nav` | 2% | 26% | 0% | 14% | 58% | 0% | 35% | 0.6 |
| `lidar` | 6% | 34% | 0% | 8% | 52% | 0% | 34% | 0.5 |
| `odom,lane,nav` | 0% | 60% | 0% | 10% | 30% | 0% | 24% | 0.3 |
