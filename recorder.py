import os
import pickle

from baselines import logger
from mpi4py import MPI
import sys


class Recorder(object):
    def __init__(self, nenvs, nlumps):
        self.nenvs = nenvs #8
        self.nlumps = nlumps #1
        self.nenvs_per_lump = nenvs // nlumps # 8//1 = 8
        self.acs = [[] for _ in range(nenvs)]
        self.int_rews = [[] for _ in range(nenvs)]
        self.ext_rews = [[] for _ in range(nenvs)]

        # add my own rewards list
        self.touch_rews = [[] for _ in range(nenvs)]
        self.obj_pos = [[] for _ in range(nenvs)]
        self.handregard = [[] for _ in range(nenvs)]
        self.action = [[] for _ in range(nenvs)]
        self.touch_F = [[] for _ in range(nenvs)]
        self.touch_S = [[] for _ in range(nenvs)]
        self.move = [[] for _ in range(nenvs)]
        self.touch_out = [[] for _ in range(nenvs)]
        self.is_done = [[] for _ in range(nenvs)]
        self.touch_all = [[] for _ in range(nenvs)]
        self.touch_fm_R = [[] for _ in range(nenvs)]
        self.touch_fm_L = [[] for _ in range(nenvs)]

        self.angle_frame = [[] for _ in range(nenvs)]
        self.obj_frame = [[] for _ in range(nenvs)]
        self.joint_frame = [[] for _ in range(nenvs)]

        self.angle_before_ac = [[] for _ in range(nenvs)]
        self.joint_before_ac = [[] for _ in range(nenvs)]
        self.obj_pos_after_ac = [[] for _ in range(nenvs)]

        self.ep_infos = [{} for _ in range(nenvs)]
        self.filenames = [self.get_filename(i) for i in range(nenvs)]
        if MPI.COMM_WORLD.Get_rank() == 0:
            logger.info("episode recordings saved to ", self.filenames[0])

    def record(self, timestep, lump, acs, infos, int_rew, ext_rew, news):
        for out_index in range(self.nenvs_per_lump):
            #sys.stdout.write("!!!" +str(self.nenvs_per_lump) + "!!!")
            in_index = out_index + lump * self.nenvs_per_lump
            #sys.stdout.write("???" +str(in_index) + "???")
            if timestep == 0:
                self.acs[in_index].append(acs[out_index])
            else:
                if self.is_first_episode_step(in_index):
                    try:
                        self.ep_infos[in_index]['random_state'] = infos[out_index]['random_state']
                    except:
                        pass

                self.int_rews[in_index].append(int_rew[out_index])
                self.ext_rews[in_index].append(ext_rew[out_index])
                self.touch_rews[in_index].append(infos[out_index]['touch'])
                #sys.stdout.write("???" + str(infos[out_index]['touch']) + "???")
                #self.obj_pos[in_index].append(infos[out_index]['obj_pos'][:])
                self.action[in_index].append(infos[out_index]['action'])
                self.move[in_index].append(infos[out_index]['move'])
                if infos[out_index].get('touch_first_arm') != None:
                    self.touch_F[in_index].append(infos[out_index]['touch_first_arm'])
                if infos[out_index].get('touch_second_arm') != None:
                    self.touch_S[in_index].append(infos[out_index]['touch_second_arm'])
                if infos[out_index].get('touch_out') != None:
                    self.touch_out[in_index].append(infos[out_index]['touch_out'])
                if infos[out_index].get('touch_all') != None:
                    self.touch_all[in_index].append(infos[out_index]['touch_all'])
                self.is_done[in_index].append(infos[out_index]['is_done'])
                if infos[out_index].get('touch_fm_R') != None:
                    self.touch_fm_R[in_index].append(infos[out_index]['touch_fm_R'])
                if infos[out_index].get('touch_fm_L') != None:
                    self.touch_fm_L[in_index].append(infos[out_index]['touch_fm_L'])


                if infos[out_index].get('obj_pos') != None:
                    self.obj_pos[in_index].append(infos[out_index]['obj_pos'][:])

                if infos[out_index].get('obj_pos_1') != None:
                    self.obj_pos[in_index].append(infos[out_index]['obj_pos_1'][:])
                    self.obj_pos[in_index].append(infos[out_index]['obj_pos_2'][:])

                info_frame = infos[out_index].get('info_per_frame')
                if info_frame != None:
                    self.angle_frame[in_index] += info_frame["arm_angles"]
                    self.obj_frame[in_index] += info_frame["object_position"]
                    self.joint_frame[in_index] += info_frame["joint_position"]

                if infos[out_index].get("arm_angle_before_action") != None:
                    self.angle_before_ac[in_index].append(infos[out_index]['arm_angle_before_action'])
                    self.joint_before_ac[in_index].append(infos[out_index]['joint_position_before_action'])
                    self.obj_pos_after_ac[in_index].append(infos[out_index]['obj_pos_after_action'])

                if news[out_index]:
                    self.ep_infos[in_index]['ret'] = infos[out_index]['episode']['r']
                    self.ep_infos[in_index]['len'] = infos[out_index]['episode']['l']
                    # self.dump_episode(in_index)

                self.acs[in_index].append(acs[out_index])

    def dump_episode(self, i, buff_rew): #function to make "".pk file
        episode = {#'acs': self.acs[i],
                   #'int_rew': self.int_rews[i],
                   'action': self.action[i],
                   'move': self.move[i],
                   'is_done': self.is_done[i],
                   'int_rew': buff_rew[i].tolist(),
                   'ext_rew': self.ext_rews[i],
                   'touch': self.touch_rews[i],
                   'obj_pos': self.obj_pos[i],
                   #'handregard': self.handregard[i],
                   'info': self.ep_infos[i]}
        if self.touch_F[i] != []:
            episode['touch_F'] = self.touch_F[i]
        if self.touch_S[i] != []:
            episode['touch_S'] = self.touch_S[i]
        if self.touch_all[i] != []:
            episode['touch_all'] = self.touch_all[i]
        if self.touch_out[i] != []:
            episode['touch_out'] = self.touch_out[i]
        if self.touch_fm_R[i] != []:
            episode['touch_fm_R'] = self.touch_fm_R[i]
        if self.touch_fm_L[i] != []:
            episode['touch_fm_L'] = self.touch_fm_L[i]

        if self.angle_frame[i] != [] and self.obj_frame[i] != [] and self.joint_frame[i] != []:
            episode["arm_angles_per_frame"] = self.angle_frame[i]
            episode["object_position_per_frame"] = self.obj_frame[i]
            episode["joint_position_per_frame"] = self.joint_frame[i]
        if self.angle_before_ac[i] != []:
            episode['arm_angle_before_action'] = self.angle_before_ac[i]
            episode['joint_position_before_action'] = self.joint_before_ac[i]
            episode['obj_pos_after_action'] = self.obj_pos_after_ac[i]

        filename = self.filenames[i]
        if self.episode_worth_saving(i):
            with open(filename, 'ab') as f:
                pickle.dump(episode, f, protocol=-1)
        self.acs[i].clear()
        self.int_rews[i].clear()
        self.ext_rews[i].clear()
        self.ep_infos[i].clear()
        self.touch_rews[i].clear()
        self.obj_pos[i].clear()
        self.handregard[i].clear()
        self.action[i].clear()
        self.move[i].clear()
        self.touch_F[i].clear()
        self.touch_S[i].clear()
        self.touch_out[i].clear()
        self.touch_all[i].clear()
        self.is_done[i].clear()

        self.angle_frame[i].clear()
        self.obj_frame[i].clear()
        self.joint_frame[i].clear()

        self.angle_before_ac[i].clear()
        self.joint_before_ac[i].clear()
        self.obj_pos_after_ac[i].clear()

    def episode_worth_saving(self, i):
        return (i == 0 and MPI.COMM_WORLD.Get_rank() == 0)# save only one env

    def is_first_episode_step(self, i):
        return len(self.int_rews[i]) == 0

    def get_filename(self, i):
        filename = os.path.join(logger.get_dir(), 'env{}_{}.pk'.format(MPI.COMM_WORLD.Get_rank(), i))
        return filename
