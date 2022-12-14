classdef modelEnv242 < rl.env.MATLABEnvironment
    %MODELENV10 Template for defining custom environment in MATLAB.    
    
    %% Properties (set properties' attributes accordingly)
    properties
        % Specify and initialize environment's necessary properties    
        useCustom = false;
        expLen = 6;
        % For result labelling
        modelInfo = "Training";
        
        
        initcon = readmatrix("modelinitcons.txt");
        foldExModel = readmatrix("modelfoldincs.txt");
        % in your modelinitcon set
        numdonors = 8;
        
        % Used to calculate next states
        % Used to find terminal states
        % Rounded to the nearest 100th
        foldExMap = [];
        initialState = [];
        
        % Going beyond this concentration will be penalized
        concentrationThreshold = 2.500000;
        
        % Experiment stops when concentration goes over this threshold
        concentrationStopThreshold = 3.5;
        
        % Reward for every 1 000 000 cells when the concentration is under
        % the threshold
        Reward1 = 1;
        
        % Penalty for every time the system is disturbed
        % Arbitrarily the concentration threshold 
        Penalty1 = -2.5;
        
        % Time step: useful for plotting
        timeStepStart = 0;
        timeStep = 0;
        
        % for plotting purposes
        Figure
        ConcentrationPlot
        
        % Initialize system state
        State = [0.5,1]
        
        % For reward conditions
        diluted = 0;
        overthreshold = false;
    end
    
    properties(Access = protected)
        % Initialize internal flag to indicate episode termination
        IsDone = false
        
        % Handle to figure
    end

    %% Necessary Methods
    methods              
        % Contructor method creates an instance of the environment
        % Change class name and constructor name accordingly
        function this = modelEnv242()
            % Initialize Observation settings
            ObservationInfo = rlNumericSpec([1 2]);
            % ObservationInfo = rlFiniteSetSpec(load("finitesetspace1.mat").states1);
            ObservationInfo.Name = 'TDP States';
            ObservationInfo.Description = 'concentration, time';
            
            % Initialize Action settings   
            % 0.04 0.08 0.12 0.16 0.2 0.24 0.28 0.32 1
            % 0.03 0.06 0.09 0.12 0.15 0.18 0.21 0.24 0.27 0.3 1
            ActionInfo = rlFiniteSetSpec([0.18 0.21 0.24 0.27 0.3 1]);
            ActionInfo.Name = 'TDP Actions';
            
            % The following line implements built-in functions of RL env
            this = this@rl.env.MATLABEnvironment(ObservationInfo,ActionInfo);
            
            % Initialize property values and pre-compute necessary values
            updateActionInfo(this);
        end
        
        % Apply system dynamics and simulates the environment with the 
        % given action for one step.
        function [Observation,Reward,IsDone,LoggedSignals] = step(this,Action)
            LoggedSignals = [];
            
            % Unpack state vector
            state = this.State;
            con = state(1);
            time = state(2);
            
            % Apply system dynamics        
            if Action<1
                con = con*Action;
                if ~isempty(this.Figure) && isvalid(this.Figure)
                    h = this.ConcentrationPlot;
                    addpoints(h,this.timeStep,con*1000000);
                    drawnow;
                end
            end
            
            % update properties related to reward
            if Action<1
                this.diluted = this.diluted + 1;
            end
            
            % calculate next state values
            time = time + 1;
            con = con * this.foldExMap(time);
            
            if con>(this.concentrationThreshold)
                this.overthreshold = true;
            end
            
            % Update system states
            Observation = zeros(1,2);
            Observation(1) = con;
            Observation(2) = time;
            this.State = Observation;
            
            % Get reward
            
            if this.overthreshold == false
                Reward = con*this.Reward1;
            else
                Reward = 5*this.concentrationThreshold-4*con*this.Reward1;
            end
            
            if con>=this.concentrationStopThreshold
                Reward = Reward + this.Penalty1;
            end
            
            if Action<1
                Reward = Reward + this.Penalty1;
            end           
            % Check terminal condition            
            IsDone = or(time == this.expLen,con>=this.concentrationStopThreshold);
            this.IsDone = IsDone;
            
            % (optional) use notifyEnvUpdated to signal that the 
            % environment has been updated (e.g. to update visualization)
            if this.timeStep==0
                this.timeStep = this.timeStep + 3;
            else
                this.timeStep = this.timeStep + 2;
            end
            notifyEnvUpdated(this);
        end
        
        % Reset environment to initial state and output initial observation
        function InitialObservation = reset(this)           
            if this.useCustom == false
                pickanumber = randi(numel(this.initcon));
                pickanothernumber = randi(this.numdonors);
                this.foldExMap = this.foldExModel(:,pickanumber+numel(this.initcon)*(pickanothernumber-1));
                % CHANGE NORMALIZATION /5
                InitialObservation = [this.initcon(pickanumber)./5000000,1];
                this.State = InitialObservation;
                this.timeStep = this.timeStepStart;
                this.diluted = 0;
                this.overthreshold = false;
                this.modelInfo = "Training";
            else
                InitialObservation = this.initialState;
                this.State = InitialObservation;
                this.timeStep = this.timeStepStart;
                this.expLen = numel(this.foldExMap);
                this.diluted = 0;
                this.overthreshold = false;
            end
            % (optional) use notifyEnvUpdated to signal that the 
            % environment has been updated (e.g. to update visualization)
            notifyEnvUpdated(this);
        end
    end
    %% Optional Methods (set methods' attributes accordingly)
    methods               
        % update the action info
        function updateActionInfo(this)
            this.ActionInfo.Elements = [0.18 0.21 0.24 0.27 0.3 1];
        end
        
        % (optional) Visualization method
        function plot(this)
            % Initiate the visualization
            this.Figure = figure('Name','Batch Simulation','Visible','on','HandleVisibility','off');
            ha = gca(this.Figure);
            ha.XLimMode = 'manual';
            ha.YLimMode = 'manual';
            ha.XLim = [0,11];
            ha.YLim = [0,this.concentrationThreshold * 1.2 * 1000000];
            xlabel(ha,'Day')
            ylabel(ha,'Cell concentration (cells/mL)')
            subtitle(ha,"Env: "+this.modelInfo)
            ha.TitleHorizontalAlignment = 'right';
            clf;
            hold(ha,'on');
            yline(ha,this.concentrationThreshold*1000000,'r--','Threshold')      
            this.ConcentrationPlot = animatedline(ha,0,0,'LineStyle','-','Color',[0, 0.4470, 0.7410]);
            clearpoints(this.ConcentrationPlot);
            % Update the visualization
            envUpdatedCallback(this)
       end
        
        % (optional) Properties validation through set methods
        function set.State(this,state)
            this.State = state;
            %notifyEnvUpdated(this);
        end
        function set.foldExMap(this,val)
            validateattributes(val,{'numeric'},{'finite','real','vector'},'','foldExMap');
            this.foldExMap = val;
        end
        function set.concentrationThreshold(this,val)
            validateattributes(val,{'numeric'},{'finite','real','scalar'},'','concentrationThreshold');
            this.concentrationThreshold = val;
        end
        function set.Reward1(this,val)
            validateattributes(val,{'numeric'},{'finite','real','scalar'},'','Reward1');
            this.Reward1 = val;
        end
    end
    
    methods (Access = protected)
        % (optional) update visualization everytime the environment is updated 
        % (notifyEnvUpdated is called)
        function envUpdatedCallback(this)          
            if ~isempty(this.Figure) && isvalid(this.Figure)
                h = this.ConcentrationPlot;
                if this.timeStep == this.timeStepStart
                    clearpoints(h);
                    ha = gca(this.Figure);
                    ha.YLim = [0,this.concentrationThreshold * 1.2 * 1000000];
                end
                addpoints(h,this.timeStep,this.State(1)*1000000);
                if this.State(1) > this.concentrationThreshold
                    ha = gca(this.Figure);
                    ha.YLim = [0,this.State(1) * 1.2 * 1000000];
                end
                % Refresh rendering in the figure window
                drawnow;                
            end
        end
    end
end
