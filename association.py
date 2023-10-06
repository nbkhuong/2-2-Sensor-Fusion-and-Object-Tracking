# ---------------------------------------------------------------------
# Project "Track 3D-Objects Over Time"
# Copyright (C) 2020, Dr. Antje Muntzinger / Dr. Andreas Haja.
#
# Purpose of this file : Data association class with single nearest neighbor association and gating based on Mahalanobis distance
#
# You should have received a copy of the Udacity license together with this program.
#
# https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013
# ----------------------------------------------------------------------
#

# imports
import numpy as np
from scipy.stats.distributions import chi2

# add project directory to python path to enable relative imports
import os
import sys
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

import misc.params as params 

class Association:
    '''Data association class with single nearest neighbor association and gating based on Mahalanobis distance'''
    def __init__(self):
        self.association_matrix = np.matrix([])
        self.unassigned_tracks = []
        self.unassigned_meas = []
        
    def associate(self, track_list, meas_list, KF):
             
        ############
        # TODO Step 3: association:
        # - replace association_matrix with the actual association matrix based on Mahalanobis distance (see below) for all tracks and all measurements
        # - update list of unassigned measurements and unassigned tracks
        ############
        self.association_matrix = np.zeros((len(track_list), len(meas_list)))
        self.unassigned_tracks = np.arange(len(track_list)).tolist()
        self.unassigned_meas = np.arange(len(meas_list)).tolist()
        for i, track in enumerate(track_list):
            for j, meas in enumerate(meas_list):
                MHD = self.MHD(track, meas, KF)
                sensor = meas.sensor
                if self.gating(MHD, sensor):
                    self.association_matrix[i, j] = MHD
                else:
                    self.association_matrix[i, j] = np.inf

        
        """self.association_matrix = np.zeros((len(track_list), len(meas_list)))
        self.unassigned_meas =  []
        self.unassigned_tracks = []
        for track_idx, track in enumerate(track_list):
            for meas_idx, measurement in enumerate(meas_list):
                distance = self.MHD(track, measurement, KF)  # Calculate Mahalanobis distance
                if not self.gating(distance, measurement.sensor):  # Check if measurement lies inside track's gate
                    self.association_matrix[track_idx, meas_idx] = float('inf')  # Set entry to infinity
                else:
                    self.association_matrix[track_idx, meas_idx] = distance  # Update entry with Mahalanobis distance

        for track_idx, row in enumerate(self.association_matrix):
            if np.isinf(row).all():  # Check if all entries in a row are infinity
                self.unassigned_tracks.append(track_idx)

        for meas_idx, col in enumerate(self.association_matrix.T):
            if np.isinf(col).all():  # Check if all entries in a column are infinity
                self.unassigned_meas.append(meas_idx)"""

        """# the following only works for at most one track and one measurement
        self.association_matrix = np.zeros((len(track_list), len(meas_list)))
        self.unassigned_meas =  list(range(len(meas_list)))
        self.unassigned_tracks = list(range(len(track_list)))

        if len(meas_list) > 0 and len(track_list) > 0:
            for track_idx, track in enumerate(track_list):
                for meas_idx, meas in enumerate(meas_list):
                    dist = self.MHD(track, meas, KF)
                    sensor = meas.sensor
                    if self.gating(dist, sensor):
                        self.association_matrix[track_idx, meas_idx] = dist
                    else:
                        self.association_matrix[track_idx, meas_idx] = np.inf"""
        
        ############
        # END student code
        ############ 
                
    def get_closest_track_and_meas(self):
        ############
        # TODO Step 3: find closest track and measurement:
        # - find minimum entry in association matrix
        # - delete row and column
        # - remove corresponding track and measurement from unassigned_tracks and unassigned_meas
        # - return this track and measurement
        ############

        # the following only works for at most one track and one measurement  
        if np.min(self.association_matrix) == np.inf:
            return np.nan, np.nan

        # get indices of minimum entry
        ij_min = np.unravel_index(np.argmin(self.association_matrix, axis=None), self.association_matrix.shape) 
        ind_track = ij_min[0]
        ind_meas = ij_min[1]

        # delete row and column for next update
        self.association_matrix = np.delete(self.association_matrix, ind_track, 0) 
        self.association_matrix = np.delete(self.association_matrix, ind_meas, 1)

        # update this track with this measurement
        update_track = self.unassigned_tracks[ind_track] 
        update_meas = self.unassigned_meas[ind_meas]

        # remove this track and measurement from list
        self.unassigned_tracks.remove(update_track) 
        self.unassigned_meas.remove(update_meas)  
            
        ############
        # END student code
        ############ 
        return update_track, update_meas     

    def gating(self, MHD, sensor): 
        ############
        # TODO Step 3: return True if measurement lies inside gate, otherwise False
        ############
        
        threshold = -1000
        if sensor.name == 'camera':
            threshold = chi2.ppf(params.gating_threshold, 1)
        if sensor.name == 'lidar':
            threshold = chi2.ppf(params.gating_threshold, 2)

        is_inside_gate = False
        if MHD <= threshold: 
            is_inside_gate = True
    
        return is_inside_gate  
        
        ############
        # END student code
        ############ 
        
    def MHD(self, track, meas, KF):
        ############
        # TODO Step 3: calculate and return Mahalanobis distance
        ############
        
        innovation = meas.z - meas.sensor.get_hx(track.x)
        H = meas.sensor.get_H(track.x)
        S = H * track.P * H.transpose() + meas.R
        S_inv = S.I
        return np.sqrt(innovation.transpose() * S_inv * innovation)
        
        ############
        # END student code
        ############ 
    
    def associate_and_update(self, manager, meas_list, KF):
        # associate measurements and tracks
        self.associate(manager.track_list, meas_list, KF)
    
        # update associated tracks with measurements
        while self.association_matrix.shape[0]>0 and self.association_matrix.shape[1]>0:
            
            # search for next association between a track and a measurement
            ind_track, ind_meas = self.get_closest_track_and_meas()
            if np.isnan(ind_track):
                print('---no more associations---')
                break
            track = manager.track_list[ind_track]
            
            # check visibility, only update tracks in fov    
            if not meas_list[0].sensor.in_fov(track.x):
                continue
            
            # Kalman update
            print('update track', track.id, 'with', meas_list[ind_meas].sensor.name, 'measurement', ind_meas)
            KF.update(track, meas_list[ind_meas])
            
            # update score and track state 
            manager.handle_updated_track(track)
            
            # save updated track
            manager.track_list[ind_track] = track
            
        # run track management 
        manager.manage_tracks(self.unassigned_tracks, self.unassigned_meas, meas_list)
        
        for track in manager.track_list:            
            print('track', track.id, 'score =', track.score)




# # ---------------------------------------------------------------------
# # Project "Track 3D-Objects Over Time"
# # Copyright (C) 2020, Dr. Antje Muntzinger / Dr. Andreas Haja.
# #
# # Purpose of this file : Data association class with single nearest neighbor association and gating based on Mahalanobis distance
# #
# # You should have received a copy of the Udacity license together with this program.
# #
# # https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013
# # ----------------------------------------------------------------------
# #

# # imports
# import numpy as np
# from scipy.stats.distributions import chi2

# # add project directory to python path to enable relative imports
# import os
# import sys
# PACKAGE_PARENT = '..'
# SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
# sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

# import misc.params as params 

# class Association:
#     '''Data association class with single nearest neighbor association and gating based on Mahalanobis distance'''
#     def __init__(self):
#         self.association_matrix = np.matrix([])
#         self.unassigned_tracks = []
#         self.unassigned_meas = []
        
#     def associate(self, track_list, meas_list, KF):
             
#         ############
#         # TODO Step 3: association:
#         # - replace association_matrix with the actual association matrix based on Mahalanobis distance (see below) for all tracks and all measurements
#         # - update list of unassigned measurements and unassigned tracks
#         ############
#         self.association_matrix = np.zeros((len(track_list), len(meas_list)))
#         self.unassigned_meas =  list(range(len(meas_list)))
#         self.unassigned_tracks = list(range(len(track_list)))

#         if len(meas_list) > 0 and len(track_list) > 0:
#             for track_idx, track in enumerate(track_list):
#                 for meas_idx, meas in enumerate(meas_list):
#                     dist = self.MHD(track, meas, KF)
#                     sensor = meas.sensor
#                     if self.gating(dist, sensor):
#                         self.association_matrix[track_idx, meas_idx] = dist
#                         self.unassigned_meas.remove(meas_idx)
#                         self.unassigned_tracks.remove(track_idx)
#                     else:
#                         self.association_matrix[track_idx, meas_idx] = np.inf
                    
#                     """# Check if the measurement is unassigned
#                     if self.association_matrix[track_idx, meas_idx] == float('inf'):
#                         self.unassigned_meas.append(meas_idx)"""

#             """if np.all(self.association_matrix[track_idx] == float('inf')):
#                 self.unassigned_tracks.append(track_idx)"""
                
#         ############
#         # END student code
#         ############ 
                
#     def get_closest_track_and_meas(self):
#         ############
#         # TODO Step 3: find closest track and measurement:
#         # - find minimum entry in association matrix
#         # - delete row and column
#         # - remove corresponding track and measurement from unassigned_tracks and unassigned_meas
#         # - return this track and measurement
#         ############
#         # Find the minimum entry in the association matrix
#         """min_value = np.nanmin(self.association_matrix)
#         if np.isnan(min_value):
#             return np.nan, np.nan

#         track_index, meas_index = np.where(self.association_matrix == min_value)

#         self.association_matrix = np.delete(self.association_matrix, track_index, axis=0)
#         self.association_matrix = np.delete(self.association_matrix, meas_index, axis=1)

#         self.unassigned_tracks.remove(track_index[0])
#         self.unassigned_meas.remove(meas_index[0])

#         return track_index[0], meas_index[0]"""

#         min_value = np.min(self.association_matrix)

#         # Find the indices of the minimum entry
#         min_indices = np.where(self.association_matrix == min_value)

#         # Get the row and column indices of the minimum entry
#         row_idx = min_indices[0][0]
#         col_idx = min_indices[1][0]

#         # Remove the corresponding row and column from the association matrix
#         self.association_matrix = np.delete(self.association_matrix, row_idx, axis=0)
#         self.association_matrix = np.delete(self.association_matrix, col_idx, axis=1)

#         # Remove the corresponding track and measurement from the unassigned_tracks and unassigned_meas lists
#         self.unassigned_tracks.pop(row_idx)
#         self.unassigned_meas.pop(col_idx)

#         # Return the association pair between track and measurement
#         if min_value == float('inf'):
#             track = np.nan
#             measurement = np.nan
#         else:
#             track = self.track_list[row_idx]
#             measurement = self.meas_list[col_idx]

#         return track, measurement



#         """ # the following only works for at most one track and one measurement  
#         A = self.association_matrix
#         if np.min(A) == np.inf:
#             return np.nan, np.nan

#         # get indices of minimum entry
#         ij_min = np.unravel_index(np.argmin(A, axis=None), A.shape) 
#         ind_track = ij_min[0]
#         ind_meas = ij_min[1]

#         # delete row and column for next update
#         A = np.delete(A, ind_track, 0) 
#         A = np.delete(A, ind_meas, 1)
#         self.association_matrix = A

#         # update this track with this measurement
#         update_track = self.unassigned_tracks[ind_track] 
#         update_meas = self.unassigned_meas[ind_meas]

#         # the following only works for at most one track and one measurement
#         #update_track = 0
#         #update_meas = 0
        
#         # remove from list
#         self.unassigned_tracks.remove(update_track) 
#         self.unassigned_meas.remove(update_meas)
#         self.association_matrix = np.matrix(A)
            
#         ############
#         # END student code
#         ############ 
#         return update_track, update_meas"""

#     def gating(self, MHD, sensor): 
#         ############
#         # TODO Step 3: return True if measurement lies inside gate, otherwise False
#         ############
#         threshold = -1000
#         if sensor.name == 'camera':
#             threshold = chi2.ppf(params.gating_threshold, 1)
#         if sensor.name == 'lidar':
#             threshold = chi2.ppf(params.gating_threshold, 2)

#         is_inside_gate = False
#         if MHD <= threshold: 
#             is_inside_gate = True
        
#         return is_inside_gate

#         ############
#         # END student code
#         ############ 
        
#     def MHD(self, track, meas, KF):
#         ############
#         # TODO Step 3: calculate and return Mahalanobis distance
#         ############
        
#         innovation = meas.z - meas.sensor.get_hx(track.x)
#         H = meas.sensor.get_H(track.x)
#         S = H * track.P * H.transpose() + meas.R
#         S_inv = S.I

#         return np.sqrt(innovation.transpose() * S_inv * innovation)
        
#         ############
#         # END student code
#         ############ 
    
#     def associate_and_update(self, manager, meas_list, KF):
#         # associate measurements and tracks
#         self.associate(manager.track_list, meas_list, KF)
    
#         # update associated tracks with measurements
#         while self.association_matrix.shape[0]>0 and self.association_matrix.shape[1]>0:
            
#             # search for next association between a track and a measurement
#             ind_track, ind_meas = self.get_closest_track_and_meas()
#             if np.isnan(ind_track):
#                 print('---no more associations---')
#                 break
#             track = manager.track_list[ind_track]
            
#             # check visibility, only update tracks in fov    
#             if not meas_list[0].sensor.in_fov(track.x):
#                 continue
            
#             # Kalman update
#             print('update track', track.id, 'with', meas_list[ind_meas].sensor.name, 'measurement', ind_meas)
#             KF.update(track, meas_list[ind_meas])
            
#             # update score and track state 
#             manager.handle_updated_track(track)
            
#             # save updated track
#             manager.track_list[ind_track] = track
            
#         # run track management 
#         manager.manage_tracks(self.unassigned_tracks, self.unassigned_meas, meas_list)
        
#         for track in manager.track_list:            
#             print('track', track.id, 'score =', track.score)