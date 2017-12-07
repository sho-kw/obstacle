import math
import numpy as np
from numpy.random import randint
from gym import Env
from gym.envs.classic_control import rendering

nan = float("nan")
class GeomContainer(rendering.Geom):
    def __init__(self, geom, collider_func=None, pos_x=0, pos_y=0, angle=0):
        rendering.Geom.__init__(self)
        self.geom = geom
        self.collider_func = collider_func
        self.collider = None
        self.pos = np.asarray([pos_x, pos_y], dtype=np.float32)
        assert self.pos.shape == (2,), 'Invalid pos-array shape'
        self.angle = angle
        self.abs_pos = np.copy(self.pos)
        self.abs_angle = self.angle
        self.trans = rendering.Transform()
        #
        self.add_attr(self.trans)
    def render(self):
        self.geom._color = self._color
        self.geom.attrs = self.attrs
        self.geom.render()
    #
    def set_pos(self, pos_x, pos_y):
        self.pos[:] = pos_x, pos_y
        self.update()
    def _move_by_xy(self, diff_x, diff_y):
        self.set_pos(self.pos[0] + diff_x, self.pos[1] + diff_y)
    def move(self, v):
        self._move_by_xy(v * np.cos(self.angle), v * np.sin(self.angle))
    #
    def set_angle(self, angle, deg=False):
        self.angle = angle if not deg else np.deg2rad(angle)
        self.update()
    def rotate(self, diff_angle, deg=False):
        self.set_angle(self.angle + diff_angle if not deg else np.deg2rad(diff_angle))
    #
    def update(self):
        self.trans.set_translation(*self.pos)
        self.trans.set_rotation(self.angle)
        #
        self.abs_pos[:] = 0
        self.abs_angle = 0
        prev_angle = 0
        for attr in reversed(self.attrs):
            if isinstance(attr, rendering.Transform):
                self.abs_pos += rotate(attr.translation, prev_angle)
                self.abs_angle += attr.rotation
                prev_angle = attr.rotation
        #
        if self.collider_func is not None:
            self.collider = self.collider_func(self.abs_pos, self.abs_angle)
    def get_geom_list(self):
        return [self]

class Sensor(GeomContainer):
    def __init__(self, geom, **kwargs):
        GeomContainer.__init__(self, geom, **kwargs)
    def detect(self, objects):
        raise NotImplementedError()

class DistanceSensor(Sensor):
    def __init__(self, geom, **kwargs):
        Sensor.__init__(self, geom, **kwargs)
        self.ray_geom = rendering.Line()
        self.ray_geom.set_color(1, 0.5, 0.5)
        self.intersection_pos = [0, 0]
        self.distance = 0
        self.max_distance = 1000
    def render(self):
        Sensor.render(self)
#        print(self.abs_pos)
        self.ray_geom.start = self.abs_pos
        self.ray_geom.end = self.abs_pos + rotate([70, 0], self.abs_angle)
        self.ray_geom.render()
    def get_geom_list(self):
        return Sensor.get_geom_list(self) + [self.ray_geom]
    def detect(self, visible_objects):
        '''
        ray = Ray(self.abs_pos, angle=self.abs_angle)
        candidates = []
        for obj in visible_objects:
            candidates.extend(obj.get_intersections(ray))
        print(len(candidates))
        if len(candidates) > 0:
            point = choose_nearest_point(candidates, ray.source)
            self.intersection_pos = [float(point.x), float(point.y)]
            diff_vector = self.abs_pos - self.intersection_pos
            self.distance = np.sqrt(np.dot(diff_vector, diff_vector))
        else:
            self.intersection_pos = self.abs_pos
            self.distance = self.max_distance
        '''
        self.intersection_pos = [0, 0]

class Robot(GeomContainer):
    def __init__(self, **kwargs):
        geom = rendering.make_circle(30)
        collider_func = None
#        collider_func = lambda pos, angle: Circle(Point(*pos), 30)
        GeomContainer.__init__(self, geom, collider_func=collider_func)
        #
        self.set_color(0, 0, 1)
        #
        sen_num = 20
        self.sensors = []
        for i in range(sen_num):
            dist_sensor = DistanceSensor(rendering.make_circle(0))
            dist_sensor.set_color(1, 0, 0)
            dist_sensor.set_pos(*(rotate([30, 0], 360 / sen_num * i, deg=True)))
            dist_sensor.set_angle(360 / sen_num * i, True)
            dist_sensor.add_attr(self.trans)
            self.sensors.append(dist_sensor)
    def render(self):
        GeomContainer.render(self)
        for sensor in self.sensors:
            sensor.render()
    def get_geom_list(self):
        return GeomContainer.get_geom_list(self) + self.sensors
    def update(self):
        GeomContainer.update(self)
        for sensor in self.sensors:
            sensor.update()
    def update_sensors(self, visible_objects):
        for sensor in self.sensors:
            sensor.detect(visible_objects)

def rotate(pos_array, angle, deg=False):
    pos_array = np.array(pos_array)
    if deg:
        angle = np.deg2rad(angle)
    c, s = np.cos(angle), np.sin(angle)
    rotation_matrix = np.array([[c, -s], [s, c]])
    return np.dot(rotation_matrix, pos_array.T).T

class ObstacleEnv(Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 50
    }
    def __init__(self):
        rob_r = 30
        obs_r, obs_num = 30, 10
        
        self.screen_width = 1000
        self.screen_height = 600
        self.state = np.zeros((obs_num + 1) * 2, dtype=np.float32)
        self.viewer = None
        self.robot = Robot()
        self.obstacles = []
        for i in range(obs_num):
            obs = GeomContainer(rendering.make_circle(obs_r))
            obs.set_color(0, 1, 0)
            self.obstacles.append(obs)
        UNIT_SQUARE = np.array([[100, 100], [100, 500], [900, 500], [900, 100]])
        self.wall =GeomContainer(rendering.make_polygon(UNIT_SQUARE))
        self.visible_object = []
        self.register_visible_object(self.wall)
        self.register_visible_object(self.robot)
        for obs in self.obstacles:
            self.register_visible_object(obs)
    def intersection(self):
        obs_r, obs_num = 30, 10
        sen_num = 20
        RS = []
        for i in range(sen_num):
            theta = math.radians(360 / sen_num * i)
            rs = np.asarray([100*math.cos(theta),100*math.sin(theta)], dtype=np.float32)
            RS.append(rs)
        ROb = []
        for j in range(obs_num):
            dx = self.robot.pos[0] - self.obstacles[j].pos[0]
            dy = self.robot.pos[1] - self.obstacles[j].pos[1]
            rob = np.asarray([dx,dy], dtype=np.float32)
            ROb.append(rob)
        distance =[]
        for i in range(sen_num):
            AX = []
            for j in range(obs_num):
                ax = (np.dot(ROb[j], RS[i])/100)
                px = (np.cross(ROb[j], RS[i])/100)
                if abs(px) <= obs_r and 0 <= ax and ax <= 100 + math.sqrt(math.pow(obs_r, 2) - math.pow(px, 2)):
                    AX.append(ax - math.sqrt(math.pow(obs_r, 2) - math.pow(px, 2)))
                else:
                    AX.append(nan)

            if RS[i][0] + self.robot.pos[0] < 100:
                l = -self.robot.pos[0] / math.cos(math.radians(0 + 360 / sen_num * i))
            elif RS[i][1] + self.robot.pos[1] < 100:
                l = -self.robot.pos[1] / math.sin(math.radians(0 + 360 / sen_num * i))
            elif RS[i][1] + self.robot.pos[1] > 500:
                l = (400 - self.robot.pos[1]) / math.sin(math.radians(0 + 360 / sen_num * i))
            else:
                l = nan

            if l != l:
                dis = np.nanmin(AX)
            elif np.nanmin(AX) != np.nanmin(AX):
                dis = np.nanmin(l)
            else:
                AX.append(l)
                dis = dis = np.nanmin(AX)
            distance.append(dis)
        if np.nanmin(distance) >= 0 :
            self.robot.set_color(0, 0, 0)
        else:
            self.robot.set_color(0, 0, 1)
        #print(distance)
    def _step(self, action):
        rob_r = 30
        obs_r, obs_num = 30, 10
        dx = []
        dy = []
        if action == 0:
            self.robot.move(3)
        elif action == 1:
            self.robot.rotate(60, deg=True)
        elif action == 2:
            self.robot.rotate(30, deg=True)
        elif action == 3:
            self.robot.rotate(-30, deg=True)
        elif action == 4:
            self.robot.rotate(-60, deg=True)
        elif action == 5:
            self.robot.move(-3)
        elif action == 6:
            self.robot.move(0) 

        self.robot.update_sensors(self.visible_object)
        self.update_state()
        self.intersection()
        #
        for i in range(obs_num):
            dx = self.robot.pos[0] - self.obstacles[i].pos[0]
            dy = self.robot.pos[1] - self.obstacles[i].pos[1]            
            done = pow(rob_r + obs_r,2) >= pow(dx,2) + pow(dy,2) 
            if done:
                reward = self.robot.pos[0]
                break
        if not done:
            reward = self.robot.pos[0]
        if self.robot.pos[0] >= 870:
            reward = self.robot.pos[0]
            done
        return self.state, reward, done, {}
    def _reset(self):
        obs_r, obs_num = 30, 10
        ROb = []
        self.robot.set_pos(150, 300)
        self.robot.set_angle(0)
        self.robot.set_color(0, 0, 1)
        for obs in self.obstacles:
            obs.set_pos(randint(100 + obs_r, 900 - obs_r), randint(100 + obs_r, 500 - obs_r))
            obs.set_angle(0)
        for i in range(obs_num):
            rob = np.asarray([self.obstacles[i].pos[0] - self.robot.pos[0], self.obstacles[i].pos[1] - self.robot.pos[1]], dtype=np.float32)
            ROb.append(rob)
        #self.ray_geom.set_color(1, 0.5, 0.5)
        self.update_state()
        return self.state
    def update_state(self):
        obs_num = 10
        self.state[0:2] = self.robot.pos
        for i in range(obs_num):
            self.state[(i+1)*2:(i+2)*2] = self.obstacles[i].pos
    def register_visible_object(self, geom_container):
        self.visible_object.extend(geom_container.get_geom_list())
    def _render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return
        if self.viewer is None:
            self.viewer = rendering.Viewer(self.screen_width, self.screen_height)
            #
            for geom in self.visible_object:
                self.viewer.add_geom(geom)
        return self.viewer.render(return_rgb_array=(mode=='rgb_array'))

def main():
    from gym.envs.registration import register
    register(
        id='Obstacle-v0',
        entry_point=ObstacleEnv,
        max_episode_steps=200,
        reward_threshold=100.0,
        )
    import gym
    env = gym.make('Obstacle-v0')
    for episode in range(5):
        step_count = 0
        state = env.reset()
        while True:
            env.render()
            if step_count != 60:
                action = 0
            else:
                action = 0
            state, reward, done, info = env.step(action)
            step_count += 1
            if done:
                print('finished episode {}, reward={}'.format(episode, reward))
                break

if __name__ == '__main__':
    main()
