#include <functional>
#include <iostream>
#include "cost.h"
#include "cmath"


using namespace std;

float goal_distance_cost(int goal_lane, int intended_lane, int final_lane, float distance_to_goal) {
    /*
    The cost increases with both the distance of intended lane from the goal
    and the distance of the final lane from the goal. The cost of being out of the 
    goal lane also becomes larger as vehicle approaches the goal.
    */
    int delta_d = 2.0*goal_lane - intended_lane - final_lane;
    float cost = 1 - exp(-(abs(delta_d)/ distance_to_goal));
    return cost; 
    
}

float inefficiency_cost(int target_speed, int intended_lane, int final_lane, vector<int> lane_speeds) {
    /*
    Cost becomes higher for trajectories with intended lane and final lane that have traffic slower than target_speed.
    */
    float speed_intended = lane_speeds[intended_lane];
    float speed_final = lane_speeds[final_lane];
    float delta_d = 2.0*target_speed - speed_intended - speed_final;
    float cost = delta_d/target_speed;
    return cost; 
}