import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.patches as patches
import matplotlib.colors as colors
from scipy.spatial.distance import cdist
import matplotlib.path as pth

class StackGeometry:
    """
    A class representing the geometry of a stack of hexagonal lattice structures.
    Contains functions to build up fibre design and convert to desired stack structure.

    Attributes:
    -----------
    nRings : int
        Number of hexagonal rings in the lattice structure
    corePitch : float
        Distance between adjacent cores in the lattice structure
    latticePoints : list
        List of lattice points in the hexagonal lattice
    coreLocations : list
        List of lattice points in the hexagonal lattice
    airholeCoords : list
        List of air hole coordinates
    clickPositions : list
        List of mouse click positions

    Methods:
    --------
    build_triangular_lattice():
        Builds a triangular lattice for the hexagonal lattice structure
    trim_lattice_to_fibre():
        Trims the triangular lattice to fit inside the hexagonal fibre
    add_cores_from_file(filepath: str):
        Reads a CSV file containing core locations and adds the core locations to the coreLocations attribute
    on_click(event):
        Appends the mouse click positions to the clickPositions attribute
    select_cores_with_mouse():
        Allows user to select core locations using mouse clicks
    make_cores_from_selection():
        Makes core locations from selected positions
    add_pcf_airholes(d_over_lambda: float):
        Adds air holes for all lattice points in the hexagonal lattice structure
    show_fibre_design():
        Plots the fibre design using Matplotlib
    choose_jacket_tube(jacket_od: int, jacket_id: int)
        Select tube for jacketing cane, outer and inner diameter in microns
    choose_stack_tube(stack_od: int, stack_id: int)
        Select tube for stack, outer and inner diameter in microns
    scale_lattice_to_stack_size()
        scales up fibre sizes using given jacket and stack tube to make a stack diagram
    show_stack_design()
        plots stack diagram
    """
    def __init__(self, number_of_rings, pitch):
        self.nRings = number_of_rings
        self.corePitch = pitch
        self.latticePoints = []
        self.coreLocations = []
        self.airholeCoords = []
        self.clickPositions = []
        self.secondaryAirholeCoords = []

    def build_triangular_lattice(self):
        # Loop through all lattice points in a triangular pattern
        for i in range(2*self.nRings +1):
            for j in range(2*self.nRings +1):
                x = j * self.corePitch
                y = i * self.corePitch * 0.8660254  # 0.8660254 is the sine of 60 degrees
                
                # Shift every other row of lattice points
                if i % 2 == 1:
                    x += self.corePitch / 2
                    
                # Append the lattice point to the list of points
                self.latticePoints.append((x, y))
        
        # Shift the lattice points so that the center is at the origin
        if self.nRings % 2 == 1:
            self.latticePoints = np.array(self.latticePoints) - np.array([self.nRings*self.corePitch +  self.corePitch / 2, self.nRings* self.corePitch * 0.8660254 ])
        else:
            self.latticePoints = np.array(self.latticePoints) - np.array([self.nRings*self.corePitch, self.nRings* self.corePitch * 0.8660254 ])

    def trim_lattice_to_fibre(self):
        # Calculate maximum x and y coordinates for the hexagon bounding the lattice points
        max_xr = (self.nRings)*self.corePitch 
        max_yr = (self.nRings+1)*self.corePitch * 0.8660254 - 0.001
        
        # Define the vertices of the hexagon
        hexagon_coord_x = np.array([max_xr, max_xr/2 , -max_xr/2, -max_xr, -max_xr/2, max_xr/2, max_xr])
        hexagon_coord_y = np.array([0, max_yr, max_yr, 0, -max_yr, -max_yr, 0])
        
        # Create a path object that represents the hexagon
        hexagon_poly_points = list(zip(hexagon_coord_x, hexagon_coord_y))
        hexagon_path = pth.Path(hexagon_poly_points)
        
        # Remove lattice points outside the hexagon
        self.latticePoints = self.latticePoints[hexagon_path.contains_points(self.latticePoints)]
        
        # Find the lattice points that are too far from the center of the hexagon
        points_to_keep = np.where(cdist(self.latticePoints, np.array([[0,0]]), 'euclidean')<self.nRings*self.corePitch-0.00001 , False, True)
        
        # Remove the lattice points that are too far from the center of the hexagon
        index_to_keep = np.invert(np.any(points_to_keep, axis=1))
        self.latticePoints = self.latticePoints[index_to_keep]


    def add_cores_from_file(self, filepath):
        # Read in *.csv file with coords in the following form:
        #% Model	|honeycomb finite realistic design.mph   |
        #% Version	|COMSOL 6.0.0.405                        |    
        #% Date	    |Apr 3 2023, 15:29                       |
        #% Table	|Table 10 - Point Evaluation 3           |    
        #    x	    |     y                                  |   
        #  -25.143	|  -2.561703144                          |
        #   ...     |    ...                                 |   

        coord_data = pd.read_csv(filepath, skiprows=4) # skip 4 lines of COMSOL preamble
        self.coreLocations = np.stack([coord_data['x'].to_numpy(),coord_data['y'].to_numpy()], axis=1)
    
    def on_click(self, event):
        self.clickPositions.append([event.xdata,event.ydata])

    def select_cores_with_mouse(self):
        self.clickPositions = []
        fig_cores = plt.figure(figsize=(6,6))
        ax_cores = fig_cores.add_subplot(111)
        ax_cores.scatter(self.latticePoints[:,0], self.latticePoints[:,1])
        fig_cores.canvas.mpl_connect('button_press_event', self.on_click)
        plt.show()

    def make_cores_from_selection(self):
        pos = np.array(self.clickPositions)
        points_to_keep = np.where(cdist(self.latticePoints, pos, 'euclidean')>2, False, True)
        self.coreLocations = self.latticePoints[np.any(points_to_keep, axis=1)]
        
    def make_alternative_airhole_from_selection(self, new_d_over_lambda):
        pos = np.array(self.clickPositions)
        points_to_keep = np.where(cdist(self.latticePoints, pos, 'euclidean')>2, False, True)
        airholes_to_drop =  np.where(cdist(self.airholeCoords, pos, 'euclidean')>2, False, True)
        self.secondaryAirholeCoords = self.latticePoints[np.any(points_to_keep, axis=1)]
        self.airholeCoords = self.airholeCoords[np.invert(np.any(airholes_to_drop, axis=1))]
        self.secondaryDOverLambda = new_d_over_lambda

    def add_pcf_airholes(self, d_over_lambda):
        # add main air holes for all lattice points 
        self.airholeCoords = self.latticePoints
        self.pcfDOverLambda = d_over_lambda

    def show_fibre_design(self):
        # Define a function to plot the fibre design using Matplotlib
        fig_circ = plt.figure(figsize=(9,5))
        axcirc = fig_circ.add_subplot(111)

        # Set the outer diameter of the fibre to be the core pitch distance
        outer_circ = self.corePitch
        core_circ_list = []
        outer_circ_list = []


        # Create a list of circles at each lattice point to represent the outer diameter of the fibre
        for _point in self.latticePoints:
            outer_circ_list.append(patches.Circle(_point, radius=outer_circ/2, ec='black'))

        # Create a list of circles at each core location to represent the core of the fibre
        for _point in self.coreLocations:
            core_circ_list.append(patches.Circle(_point, radius=outer_circ/2, ec='black', fc='red'))

        # Add the circles to the plot
        for _circ in outer_circ_list:
            axcirc.add_patch(_circ)

        # optionally add airholes if making pcf
        if len(self.airholeCoords) != 0:
            for _point in self.airholeCoords:
                axcirc.add_patch(patches.Circle(_point, radius=self.pcfDOverLambda*outer_circ/2, fc='white', ec='black'))
            if len(self.secondaryAirholeCoords) != 0:
                for _point in self.secondaryAirholeCoords:
                    axcirc.add_patch(patches.Circle(_point, radius=self.secondaryDOverLambda*outer_circ/2, fc='white', ec='black'))
        
        # add red circles for cores 
        for _circ in core_circ_list:
            axcirc.add_patch(_circ)

        # Set the axis to be scaled and show the plot
        plt.axis('scaled')
        plt.show()

    def choose_jacket_tube(self, jacket_od, jacket_id):
        # Define a function to set the outer and inner diameters of the jacket tube
        # Use microns for all sizes
        self.jacketId = jacket_id
        self.jacketOd = jacket_od

    def choose_stack_tube(self, stack_od, stack_id):
        # Define a function to set the outer and inner diameters of the stack tube
        # Use microns for all sizes
        self.stackId = stack_id
        self.stackOd = stack_od 

        # Calculate the required capillary diameter based on the largest stack diameter once corners have been dropped
        self.capillaryDiameter = float(self.stackId)/np.sqrt((2*self.nRings)**2 + 3)
        print('Capillary Diameter Required: {}'.format(self.capillaryDiameter))

    def scale_lattice_to_stack_size(self):
        # Define a function to scale the lattice points and core locations to the size of the stack
        scale_factor_stack_to_cane = self.jacketId/self.stackOd
        scale_factor_cap_to_pitch = self.corePitch/self.capillaryDiameter
        scale_factor_cane_to_fib = scale_factor_cap_to_pitch/scale_factor_stack_to_cane

        # Print the required fibre diameter
        print('Required Fibre Diamter: {}'.format(self.jacketOd*scale_factor_cane_to_fib))

        # Scale the core locations and lattice points and airhole locations
        self.stackLatticePoints = self.latticePoints/scale_factor_cap_to_pitch

        if len(self.airholeCoords) != 0:
            self.airholeCoords = self.airholeCoords/scale_factor_cap_to_pitch
            if len(self.secondaryAirholeCoords) != 0:
                self.secondaryAirholeCoords = self.secondaryAirholeCoords/scale_factor_cap_to_pitch

        if len(self.coreLocations) != 0:
            self.stackCoreLocations = self.coreLocations/scale_factor_cap_to_pitch


    def show_stack_design(self):
        # Create a new figure with a size of 9 inches by 5 inches
        fig_circ = plt.figure(figsize=(9,5))
        # Add a subplot to the figure
        axcirc = fig_circ.add_subplot(111)

        # Create empty lists for circles representing the stack fibers and jacket
        stack_core_circ_list = []
        stack_outer_circ_list = []
        
        # Add a black circle representing the outer diameter of the jacket
        axcirc.add_patch(patches.Circle((0, 0), radius=self.stackOd/2, ec='black'))
        # Add a white circle representing the inner diameter of the jacket
        axcirc.add_patch(patches.Circle((0, 0), radius=self.stackId/2, color='white', ec='black'))

        # Loop through the lattice points of the stack fibers and add circles to the list
        for _point in self.stackLatticePoints:
            stack_outer_circ_list.append(patches.Circle(_point, radius=self.capillaryDiameter/2, ec='black'))

        # Loop through the core locations of the stack fibers and add circles to the list
        for _point in self.stackCoreLocations:
            stack_core_circ_list.append(patches.Circle(_point, radius=self.capillaryDiameter/2, ec='black', fc='red'))
            
        # Add the circles for the stack fibers to the plot
        for _circ in stack_outer_circ_list:
            axcirc.add_patch(_circ)

        # optionally add airholes if making pcf
        if len(self.airholeCoords) != 0:
            for _point in self.airholeCoords:
                axcirc.add_patch(patches.Circle(_point, radius=self.pcfDOverLambda*self.capillaryDiameter/2, fc='white', ec='black')) 
            if len(self.secondaryAirholeCoords) != 0:
                for _point in self.secondaryAirholeCoords:
                    axcirc.add_patch(patches.Circle(_point, radius=self.secondaryDOverLambda*self.capillaryDiameter/2, fc='white', ec='black'))
        # Add the circles for the core locations of the stack fibers to the plot
        for _circ in stack_core_circ_list:
            axcirc.add_patch(_circ)
        self.stackDesign = fig_circ
        # Set the aspect ratio of the plot to 'equal'
        plt.axis('scaled')
        # Show the plot
        plt.show()

    def export_stack_design(self, filename):
        self.stackDesign.savefig('{}.svg'.format(filename), dpi=300)

class GlassChecker:
    """
    A class for calculating and displaying the required parameters for making optical fibers.

    Attributes:
        dOverLambda (float): The value of the ratio of the fiber diameter to the wavelength of the light.
        desiredPitch (float): The desired pitch of the fiber cores in micrometers.
        goalDiameter (float): The ideal diameter of the optical fiber in micrometers.
        nRings (int): The number of rings in the fiber cladding.
        nCores (int): The number of cores in the fiber.
        nCapsNotCores (int): The number of capillaries required for the fiber cladding (excluding those used for the cores).

    Methods:
        read_latest_tube_list(filepath): Reads an Excel file and sets the list of available tubes.
        show_closest_capillary_tubes(): Displays a table of the capillary tubes that are closest to the required diameter.
        show_available_stack_tubes(): Displays a table of the available stack tubes.
        set_capillary_sizes(capillary_tube_od, capillary_tube_id, core_cap_od, core_cap_id): Sets the sizes of the capillary tubes and core caps.
        set_stack_size(stack_od, stack_id): Sets the sizes of the stack tubes.
        calc_capillary_diameter(): Calculates the required diameter of the capillary tubes and displays the necessary lengths of capillary and core tubes.
        show_jacket_options(): Displays a table of the available jacket tubes and their relevant parameters.
    """
    def __init__(self, d_over_lambda, desired_pitch, ideal_fibre_diameter, number_of_rings, number_of_cores):
        """
        Initializes the GlassChecker class.

        Args:
            d_over_lambda (float): The value of the ratio of the fiber diameter to the wavelength of the light.
            desired_pitch (float): The desired pitch of the fiber cores in micrometers.
            ideal_fibre_diameter (float): The ideal diameter of the optical fiber in micrometers.
            number_of_rings (int): The number of rings in the fiber cladding.
            number_of_cores (int): The number of cores in the fiber.
        """
        self.dOverLambda = d_over_lambda
        self.desiredPitch = desired_pitch
        self.goalDiameter =  ideal_fibre_diameter
        self.nRings = number_of_rings
        self.nCores = number_of_cores
        self.nCapsNotCores = 6*sum([i for i in range(self.nRings+1)]) - self.nCores

    def read_latest_tube_list(self, filepath):
        """
        Reads the tube list Excel file and sets the list of available tubes.
        Splits tubes into stack and jacket for easier separation later

        Args:
            filepath (str): The file path of the Excel file.
        """
        latest_excel_sheet = pd.read_excel(filepath, sheet_name=1)
        is_available = latest_excel_sheet['Available'] == 'Y'
        self.availableTubeList = latest_excel_sheet[is_available]
        self.stackTubes = self.availableTubeList[self.availableTubeList['NomOD'].astype('float64') > 20]
        self.jacketTubes = self.availableTubeList[self.availableTubeList['NomOD'].astype('float64') < 12]
    
    def show_closest_capillary_tubes(self):
        """
        Displays a table of the capillary tubes that are closest to the required diameter.
        """
        closest_caps = self.availableTubeList.iloc[(self.availableTubeList['ID/OD']-self.dOverLambda).abs().argsort()[:20]]
        closest_caps = closest_caps[['Tube No','NomOD', 'NomID', 'ID/OD', 'Od Avg', 'Id Avg','id/od']]
        display(closest_caps.head(10))
    
    def show_available_stack_tubes(self):
        """
        Displays a table of the available stack tubes.
        """
        stack_tubes_to_show = self.stackTubes[['Tube No','NomOD', 'NomID', 'ID/OD', 'Od Avg', 'Id Avg','id/od']]
        display(stack_tubes_to_show.head(10))

    def set_capillary_sizes(self, capillary_tube_od, capillary_tube_id, core_cap_od, core_cap_id):
        """
        Set the sizes of the capillary tubes and core tubes.

        Parameters:
        capillary_tube_od (float): outer diameter of the capillary tubes.
        capillary_tube_id (float): inner diameter of the capillary tubes.
        core_cap_od (float): outer diameter of the core tubes.
        core_cap_id (float): inner diameter of the core tubes.
        """
        self.capillaryOdTube = capillary_tube_od
        self.capillaryIdTube = capillary_tube_id
        self.coreCapOd = core_cap_od
        self.coreCapId = core_cap_id


    def set_stack_size(self, stack_od, stack_id):
        """
        Set the sizes of the stack.

        Parameters:
        stack_od (float): outer diameter of the stack.
        stack_id (float): inner diameter of the stack.
        """
        self.stackOd = stack_od
        self.stackId = stack_id


    def calc_capillary_diameter(self):
        """
        Calculate the diameter of the capillary tubes needed for the given stack and number of layers.

        Prints the required capillary diameter, the length of capillary tube needed for caps, and the length of
        capillary tube needed for cores.
        """
        self.final_capillary_diameter = self.stackId/np.sqrt(3+(2*self.nRings)**2)
        length_of_cap_tube_needed_caps = (self.final_capillary_diameter/self.capillaryOdTube)**2 * self.nCapsNotCores
        length_of_cap_tube_needed_cores = (self.final_capillary_diameter/self.coreCapOd)**2 * self.nCores
        print('Required Capillary Diameter: {}'.format(self.final_capillary_diameter))
        print('Length of Core tube needed: {}'.format(length_of_cap_tube_needed_cores))
        print('Length of Capillary tube needed: {}'.format(length_of_cap_tube_needed_caps))


    def show_jacket_options(self):
        """
        Show the different jacket options available and the corresponding fibre parameters.

        Prints a table with the fibre pitch at 150um, the fibre diameter for ideal pitch, the number of jacket tubes,
        the jacket OD, and the jacket ID for each jacket ratio.
        """
        jacket_ratio_list = np.unique(self.jacketTubes['ID/OD'])
        jacket_dict = {b: len(self.jacketTubes[self.jacketTubes['ID/OD'] == b]) for b in jacket_ratio_list}

        fibre_param_options = pd.DataFrame(columns=['Fibre Pitch at 150um (um)','Fibre Diameter for Ideal Pitch (um)',
                                                    'Number of Jacket Tubes', 'Jacket Od (mm)', 'Jacket Id (mm)'])

        for key in list(jacket_dict.keys()):
            useable_jacket_tubes = self.jacketTubes[self.jacketTubes['ID/OD'] == key][['NomOD','NomID']]
            jacket_od = useable_jacket_tubes.iloc[0]['NomOD']
            jacket_id = useable_jacket_tubes.iloc[0]['NomID']

            # calculate fibre pitch and necessary fibre OD
            rcf = jacket_od/self.goalDiameter
            rsc = self.stackOd/jacket_id
            fibre_pitch = self.final_capillary_diameter/(rsc*rcf)
            necessary_fibre_od = rsc*jacket_od*self.desiredPitch/self.final_capillary_diameter

            # add results to dataframe
            fibre_param_options = fibre_param_options.append(pd.DataFrame([[fibre_pitch, necessary_fibre_od,
                                                                            jacket_dict[key], jacket_od, jacket_id]],
                                                                            columns=['Fibre Pitch at 150um (um)','Fibre Diameter for Ideal Pitch (um)',
                                                                            'Number of Jacket Tubes', 'Jacket Od (mm)', 'Jacket Id (mm)']),
                                                                            ignore_index=True)
        display(fibre_param_options)