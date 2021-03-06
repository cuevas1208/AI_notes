#include <fstream>
#include <math.h>
#include <uWS/uWS.h>
#include <chrono>
#include <iostream>
#include <thread>
#include <vector>
#include "Eigen-3.3/Eigen/Core"
#include "Eigen-3.3/Eigen/QR"
#include "json.hpp"
#include "spline.h"

using namespace std;

// for convenience
using json = nlohmann::json;

//reference velocity
double max_speed = 49.5;
double changing_lane_speed = 45;
double max_acceleration = .224;
int ROAD_LANES = 3;

double ref_vel = 1;
int goal_lane = 1;
int confirm_state = 0;
int changing_lanes_delay = 50;
bool ok_to_changing_lanes = false;

// For converting back and forth between radians and degrees.
constexpr double pi() { return M_PI; }
double deg2rad(double x) { return x * pi() / 180; }
double rad2deg(double x) { return x * 180 / pi(); }

// Checks if the SocketIO event has JSON data.
// If there is data the JSON object in string format will be returned,
// else the empty string "" will be returned.
string hasData(string s) {
  auto found_null = s.find("null");
  auto b1 = s.find_first_of("[");
  auto b2 = s.find_first_of("}");
  if (found_null != string::npos) {
    return "";
  } else if (b1 != string::npos && b2 != string::npos) {
    return s.substr(b1, b2 - b1 + 2);
  }
  return "";
}

double distance(double x1, double y1, double x2, double y2)
{
  return sqrt((x2-x1)*(x2-x1)+(y2-y1)*(y2-y1));
}

//different from next point this maybe behind the card
int ClosestWaypoint(double x, double y, const vector<double> &maps_x, const vector<double> &maps_y)
{

  double closestLen = 100000; //large number
  int closestWaypoint = 0;

  for(int i = 0; i < maps_x.size(); i++)
  {
    double map_x = maps_x[i];
    double map_y = maps_y[i];
    double dist = distance(x,y,map_x,map_y);
    if(dist < closestLen)
    {
      closestLen = dist;
      closestWaypoint = i;
    }
  }

  return closestWaypoint;

}

int NextWaypoint(double x, double y, double theta, const vector<double> &maps_x, const vector<double> &maps_y)
{

  int closestWaypoint = ClosestWaypoint(x,y,maps_x,maps_y);

  double map_x = maps_x[closestWaypoint];
  double map_y = maps_y[closestWaypoint];

  double heading = atan2((map_y-y),(map_x-x));

  double angle = fabs(theta-heading);
  angle = min(2*pi() - angle, angle);

  if(angle > pi()/4)
  {
    closestWaypoint++;
  if (closestWaypoint == maps_x.size())
  {
    closestWaypoint = 0;
  }
  }

  return closestWaypoint;
}

// Transform from Cartesian x,y coordinates to Frenetic s,d coordinates
vector<double> getFrenet(double x, double y, double theta, const vector<double> &maps_x, const vector<double> &maps_y)
{
  int next_wp = NextWaypoint(x,y, theta, maps_x,maps_y);

  int prev_wp;
  prev_wp = next_wp-1;
  if(next_wp == 0)
  {
    prev_wp  = maps_x.size()-1;
  }

  double n_x = maps_x[next_wp]-maps_x[prev_wp];
  double n_y = maps_y[next_wp]-maps_y[prev_wp];
  double x_x = x - maps_x[prev_wp];
  double x_y = y - maps_y[prev_wp];

  // find the projection of x onto n
  double proj_norm = (x_x*n_x+x_y*n_y)/(n_x*n_x+n_y*n_y);
  double proj_x = proj_norm*n_x;
  double proj_y = proj_norm*n_y;

  double frenet_d = distance(x_x,x_y,proj_x,proj_y);

  //see if d value is positive or negative by comparing it to a center point

  double center_x = 1000-maps_x[prev_wp];
  double center_y = 2000-maps_y[prev_wp];
  double centerToPos = distance(center_x,center_y,x_x,x_y);
  double centerToRef = distance(center_x,center_y,proj_x,proj_y);

  if(centerToPos <= centerToRef)
  {
    frenet_d *= -1;
  }

  // calculate s value
  double frenet_s = 0;
  for(int i = 0; i < prev_wp; i++)
  {
    frenet_s += distance(maps_x[i],maps_y[i],maps_x[i+1],maps_y[i+1]);
  }

  frenet_s += distance(0,0,proj_x,proj_y);

  return {frenet_s,frenet_d};

}

// Transform from Frenet s,d coordinates to Cartesian x,y
vector<double> getXY(double s, double d, const vector<double> &maps_s, const vector<double> &maps_x, const vector<double> &maps_y)
{
  int prev_wp = -1;

  while(s > maps_s[prev_wp+1] && (prev_wp < (int)(maps_s.size()-1) ))
  {
    prev_wp++;
  }

  int wp2 = (prev_wp+1)%maps_x.size();

  double heading = atan2((maps_y[wp2]-maps_y[prev_wp]),(maps_x[wp2]-maps_x[prev_wp]));
  // the x,y,s along the segment
  double seg_s = (s-maps_s[prev_wp]);

  double seg_x = maps_x[prev_wp]+seg_s*cos(heading);
  double seg_y = maps_y[prev_wp]+seg_s*sin(heading);

  double perp_heading = heading-pi()/2;

  double x = seg_x + d*cos(perp_heading);
  double y = seg_y + d*sin(perp_heading);

  return {x,y};

}

int main() {
  uWS::Hub h;

  // Load up map values for waypoint's x,y,s and d normalized normal vectors
  vector<double> map_waypoints_x;
  vector<double> map_waypoints_y;
  vector<double> map_waypoints_s;
  vector<double> map_waypoints_dx;
  vector<double> map_waypoints_dy;

  // Waypoint map to read from
  string map_file_ = "../data/highway_map.csv";
  // The max s value before wrapping around the track back to 0
  double max_s = 6945.554;

  ifstream in_map_(map_file_.c_str(), ifstream::in);

  string line;
  while (getline(in_map_, line)) {
    istringstream iss(line);
    double x;
    double y;
    float s;
    float d_x;
    float d_y;
    iss >> x;
    iss >> y;
    iss >> s;
    iss >> d_x;
    iss >> d_y;
    map_waypoints_x.push_back(x);
    map_waypoints_y.push_back(y);
    map_waypoints_s.push_back(s);
    map_waypoints_dx.push_back(d_x);
    map_waypoints_dy.push_back(d_y);
  }

  h.onMessage([&map_waypoints_x,&map_waypoints_y,&map_waypoints_s,&map_waypoints_dx,&map_waypoints_dy](uWS::WebSocket<uWS::SERVER> ws, char *data, size_t length,
                     uWS::OpCode opCode) {
    // "42" at the start of the message means there's a websocket message event.
    // The 4 signifies a websocket message
    // The 2 signifies a websocket event
    //auto sdata = string(data).substr(0, length);
    //cout << sdata << endl;
    if (length && length > 2 && data[0] == '4' && data[1] == '2') {

      auto s = hasData(data);

      if (s != "") {
        auto j = json::parse(s);

        string event = j[0].get<string>();

        if (event == "telemetry") {
          // j[1] is the data JSON object

          // Main car's localization Data
            double my_car_x = j[1]["x"];
            double my_car_y = j[1]["y"];
            double my_track_distance = j[1]["s"];
            double my_car_raw_lane = j[1]["d"];
            double my_car_yaw = j[1]["yaw"];
            double my_speed = j[1]["speed"];

            // Previous path = the dots that the car still see in front
            // from 30 pts, if the car moved 3 pts, previous path would still have 27
            auto previous_path_x = j[1]["previous_path_x"];
            auto previous_path_y = j[1]["previous_path_y"];

            // Previous path's end s and d values
            double end_path_s = j[1]["end_path_s"];
            double end_path_d = j[1]["end_path_d"];

            // Sensor Fusion Data, a list of all other cars on the same side of the road.
            // this is how sensor fusion looks like  [ id, x, y, vx, vy, s, d]
            auto sensor_fusion = j[1]["sensor_fusion"];
            int prev_size = previous_path_x.size();

            json msgJson;

            //create a list of 30m spaced (x,y) waypoints
            vector<double> ptsx;
            vector<double> ptsy;

            int my_car_lane = int((((my_car_raw_lane - 2)/4) + ((my_car_raw_lane + 2) /4))/2);

            bool too_close = false;
            bool free_lane[3] = {true, true, true};

            double goal_speed = max_speed;
            int lane_car_distance[3] = {1000, 1000, 1000};
            double front_car_speed[3] = {max_speed, max_speed, max_speed};

            if(my_car_lane == goal_lane){
                if (!ok_to_changing_lanes)
                    changing_lanes_delay--;
                if (changing_lanes_delay == 0)
                    ok_to_changing_lanes = true;
            }
            else
                ok_to_changing_lanes = false;


            // iterate every car detected in the sensor fusion
            for(int i = 0; i< sensor_fusion.size(); i++)
            {
                // get car's lane
                float sf_car_raw_lane = sensor_fusion[i][6];
                int sf_car_lane = int((sf_car_raw_lane - 2) /4);

                // if car is within ROAD_LIMIT
                if(sf_car_lane <= ROAD_LANES && sf_car_lane >= 0){
                    double vx = sensor_fusion[i][3];                            // x velocity
                    double vy = sensor_fusion[i][4];                            // y velocity
                    double sf_car_speed = sqrt(vx*vx+vy*vy);                    // calculate cars speed
                    double sf_car_track_distance = sensor_fusion[i][5];         // get car distance in the track
                    // sf_car_track_distance += prev_size * .02 * sf_car_speed;

                    int cars_distance = sf_car_track_distance - my_track_distance;

                    bool car_is_front = sf_car_track_distance > my_track_distance;

                    if (lane_car_distance[sf_car_lane] > cars_distance && car_is_front){
                       lane_car_distance[sf_car_lane] = cars_distance;
                       front_car_speed[sf_car_lane] = sf_car_speed;
                    }

                    if(sf_car_lane == my_car_lane){

                       // if car is in front is less than 30 set too_close flag
                       if(car_is_front){
                            if(cars_distance < 30)
                                too_close = true;

                            if (cars_distance < 50)
                               free_lane[sf_car_lane] = false;
                       }
                    }

                    // if lane is around my car lanes
                    if (ok_to_changing_lanes)
                        if (sf_car_lane == my_car_lane+1 || sf_car_lane == my_car_lane-1){
                            //check to see if they are any cars with in +/- 20 meters
                            if ((cars_distance > -25) && (cars_distance < 25))
                                free_lane[sf_car_lane] = false;
                        }
                }
            }

			int want_change_lanes = 0;
            for (int i =2; i > -1; i--){
                cout << lane_car_distance[i] << " - " ;
            }
            cout << changing_lanes_delay << "-" << my_car_lane << " - " << goal_lane << " - " << endl;

            // accelerate decision
            if (too_close){
               //cout << "\n FRONT CAR DISTANCE: " << front_car_distance ;
               //cout << " SPEED: " << front_car_speed << endl;
               if (front_car_speed[my_car_lane] < goal_speed)
                    goal_speed = front_car_speed[my_car_lane] + 2;
            } else
                goal_speed = max_speed;

            // change lane decision
            if (!free_lane[my_car_lane]){
                    if(ok_to_changing_lanes){
                        if (my_speed < changing_lane_speed){
                            want_change_lanes = 1;
                            // find the best lane
                            const int N = sizeof(lane_car_distance) / sizeof(int);
                            int lane_to_go_by_distance = distance(lane_car_distance, max_element(lane_car_distance, lane_car_distance + N));

                            const int M = sizeof(front_car_speed) / sizeof(double);
                            int lane_to_go_by_speed = distance(front_car_speed, max_element(front_car_speed, front_car_speed + M));

                            if (lane_to_go_by_distance == lane_to_go_by_speed){
                                int lane_to_go = lane_to_go_by_distance;

                                // change lane one at the time
                                if (lane_to_go > my_car_lane)
                                    lane_to_go = my_car_lane+1;
                                else if (lane_to_go < my_car_lane)
                                    lane_to_go = my_car_lane-1;

                                //change lane if lane is free of traffic
                                if (free_lane[lane_to_go]){
                                    goal_lane = lane_to_go;
                                    cout << "changing lane to" << lane_to_go << endl;
                                    changing_lanes_delay = 50;
                                }
                            }
                        }else
                            goal_speed--;
                    }
            }


            float acceleration = max_acceleration*((goal_speed - ref_vel)/goal_speed);
            ref_vel += acceleration;

            double ref_x = my_car_x;
            double ref_y = my_car_y;
            double ref_yaw = deg2rad(my_car_yaw);

            //if previous size is almost empty, use the car as starting point
            if (prev_size < 2)
            {
              //use two points that make the path tangent to the car
              double prev_my_car_x = my_car_x - cos(my_car_yaw);
              double prev_car_y = my_car_y - sin(my_car_yaw);

              ptsx.push_back(prev_my_car_x);
              ptsx.push_back(my_car_x);

              ptsy.push_back(prev_car_y);
              ptsy.push_back(my_car_y);
            }
            else
            { //reference data as previews path
              ref_x = previous_path_x[prev_size-1];
              ref_y = previous_path_y[prev_size-1];

              double ref_x_prev = previous_path_x[prev_size-2];
              double ref_y_prev = previous_path_y[prev_size-2];

              //calculate the angle of the last two dots in the previews points
              ref_yaw = atan2(ref_y - ref_y_prev, ref_x - ref_x_prev);

              //use two points that make the path tangent to the previous path's
              ptsx.push_back(ref_x_prev);
              ptsx.push_back(ref_x);

              ptsy.push_back(ref_y_prev);
              ptsy.push_back(ref_y);

            }

            // set how far to plan for the target the longer the smother
            double target_x = 50.0;

            // TODO: define a path made up of (x,y) points that the car will visit sequentially every .02 seconds
            // 30 * .02 = .6 sec
            for(int i = 1; i < 4; i++)
            {
              // in frenet add evenly every 30m space points
              vector<double> next_wp = getXY(my_track_distance+(i*target_x),(2+4*goal_lane), map_waypoints_s,
                map_waypoints_x,map_waypoints_y);
              ptsx.push_back(next_wp[0]);
              ptsy.push_back(next_wp[1]);
            }


            //shift car reference angle to 0 degrees
            for (int i = 0; i < ptsy.size(); i++)
            {
              double shift_x = ptsx[i]-ref_x;
              double shift_y = ptsy[i]-ref_y;

              ptsx[i] = (shift_x*cos(0-ref_yaw)-shift_x*sin(0-ref_yaw));
              ptsy[i] = (shift_x*sin(0-ref_yaw)+shift_y*cos(0-ref_yaw));
            }

            // create a spline, spline would help us navigate at reasonable speed
            tk::spline s;

            //set points to spline
            s.set_points(ptsx, ptsy);

            //set(x,y) points we will use for the planner
            std::vector<double> next_x_vals;
            std::vector<double> next_y_vals;

            //re-used previews path available
            for(int i = 0; i < previous_path_x.size(); i++)
            {
              next_x_vals.push_back(previous_path_x[i]);
              next_y_vals.push_back(previous_path_y[i]);
            }

            //calculate how to break up spline points so that we set velocity
            target_x = 30;
            double target_y = s(target_x);
            double target_dist = sqrt((target_x)*(target_x)+(target_y)*(target_y));

            double x_add_on = 0;

            //0.2 / ref_vel/2.24 should give us the distance for the next point
            //2.24 would convert from miles/hour to m/s
            double N = (target_dist/(.02 * ref_vel/2.24));

            //create the new points path
            for (int i = 1; i <= 60 - previous_path_x.size(); i++){
              double x_point = x_add_on+(target_x)/N;
              double y_point = s(x_point);

              x_add_on = x_point;

              double x_ref = x_point;
              double y_ref = y_point;

              // rotate back to normal after rotating
              x_point = (x_ref*cos(ref_yaw)-y_ref*sin(ref_yaw));
              y_point = (x_ref*sin(ref_yaw)+y_ref*cos(ref_yaw));

              x_point += ref_x;
              y_point += ref_y;

              next_x_vals.push_back(x_point);
              next_y_vals.push_back(y_point);

            }

            msgJson["next_x"] = next_x_vals;
            msgJson["next_y"] = next_y_vals;

            auto msg = "42[\"control\","+ msgJson.dump()+"]";

            //this_thread::sleep_for(chronic::milliseconds(1000));
            ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);

        }
      } else {
        // Manual driving
        std::string msg = "42[\"manual\",{}]";
        ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
      }
    }
  });

  // We don't need this since we're not using HTTP but if it's removed the
  // program
  // doesn't compile :-(
  h.onHttpRequest([](uWS::HttpResponse *res, uWS::HttpRequest req, char *data,
                     size_t, size_t) {
    const std::string s = "<h1>Hello world!</h1>";
    if (req.getUrl().valueLength == 1) {
      res->end(s.data(), s.length());
    } else {
      // i guess this should be done more gracefully?
      res->end(nullptr, 0);
    }
  });

  h.onConnection([&h](uWS::WebSocket<uWS::SERVER> ws, uWS::HttpRequest req) {
    std::cout << "Connected!!!" << std::endl;
  });

  h.onDisconnection([&h](uWS::WebSocket<uWS::SERVER> ws, int code,
                         char *message, size_t length) {
    ws.close();
    std::cout << "Disconnected" << std::endl;
  });

  int port = 4567;
  if (h.listen(port)) {
    std::cout << "Listening to port " << port << std::endl;
  } else {
    std::cerr << "Failed to listen to port" << std::endl;
    return -1;
  }
  h.run();
}
