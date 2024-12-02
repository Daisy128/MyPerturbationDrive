from tensorflow.keras.models import load_model
import numpy as np
import os

from .action import UdacityAction
from .observation import UdacityObservation
from .pre_process import preprocess

class UdacityAgent:

    def __init__(self, before_action_callbacks=None, after_action_callbacks=None, transform_callbacks=None):
        self.before_action_callbacks = before_action_callbacks if before_action_callbacks is not None else []
        self.after_action_callbacks = after_action_callbacks if after_action_callbacks is not None else []
        self.transform_callbacks = transform_callbacks if transform_callbacks is not None else []

    def on_before_action(self, observation: UdacityObservation, *args, **kwargs):
        for callback in self.before_action_callbacks:
            callback(observation, *args, **kwargs)

    def on_after_action(self, observation: UdacityObservation, *args, **kwargs):
        for callback in self.after_action_callbacks:
            callback(observation, *args, **kwargs)

    def on_transform_observation(self, observation: UdacityObservation, *args, **kwargs):
        for callback in self.transform_callbacks:
            observation = callback(observation, *args, **kwargs)
        return observation

    def action(self, observation: UdacityObservation, *args, **kwargs):
        raise NotImplementedError('UdacityAgent does not implement __call__')

    def __call__(self, observation: UdacityObservation, *args, **kwargs):
        if observation.input_image is None:
            return UdacityAction(steering_angle=0.0, throttle=0.0)
        self.on_before_action(observation)
        observation = self.on_transform_observation(observation)
        action = self.action(observation, *args, **kwargs)
        self.on_after_action(observation, action=action)
        return action

class SupervisedAgent(UdacityAgent):

    def __init__(
            self,
            model_path: str,
            max_speed: int,
            min_speed: int,
            predict_throttle: bool = False,
    ):
        super().__init__(before_action_callbacks=None, after_action_callbacks=None)

        # assert检查模型路径是否存在，不存在抛出错误信息
        assert os.path.exists(model_path), 'Model path {} not found'.format(model_path)

        self.model = load_model(model_path)
        self.predict_throttle = predict_throttle
        self.max_speed = max_speed
        self.min_speed = min_speed

    def action(self, observation: UdacityObservation, *args, **kwargs) -> UdacityAction:
        # observation by getting coordinate each time
        obs = observation.input_image # batch of images

        #print("Observations:", obs)
        obs = preprocess(obs)
        
        #  the model expects 4D array
        obs = np.array([obs])

        # obs = torch.transforms.Normalize(obs_mean,obs_std)
        speed = observation.speed
    
        if self.predict_throttle:
            action = self.model.predict(obs, batch_size=1, verbose=0)
            steering, throttle = action[0][0], action[0][1]
        else:
            import time
            time_start = time.time()
            steering = float(self.model.predict(obs, batch_size=1, verbose=0)[0])
            #print("DNN elasped time ",time.time() - time_start)
            steering = np.clip(steering, -1, 1)
            if speed > self.max_speed:
                speed_limit = self.min_speed  # slow down
            else:
                speed_limit = self.max_speed
            
            #steering = self.change_steering(steering=steering)
            #steering = float(self.model.predict(obs, batch_size=1, verbose=0))

            throttle = np.clip(a=1.0 - steering ** 2 - (speed / speed_limit) ** 2, a_min=0.0, a_max=1.0)

            #print(f"steering {steering} throttle {throttle}")
            #self.model.summary()

        return UdacityAction(steering_angle=steering, throttle=throttle)
