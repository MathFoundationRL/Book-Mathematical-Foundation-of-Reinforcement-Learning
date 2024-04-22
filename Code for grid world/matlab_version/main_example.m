% by the Intelligent Unmanned Systems Laboratory, Westlake University, 2024

clear 
close all

% Initialize environment parameters
agent_state = [1, 1];
final_state = [3, 3];
obstacle_state = [1, 3; 2, 1; 1, 2];
x_length = 3;
y_length = 4;
state_space = x_length * y_length; 
state=1:state_space;
state_value=ones(state_space,1);

reward_forbidden = -1;
reward_target = 1;
reward_step = 0;

% Define actions: up, right, down, left, stay
actions = {[0, -1], [1, 0], [0, 1], [-1, 0], [0, 0]};

% Initialize a cell array to store the action space for each state
action_space = cell(state_space, 1);

% Populate the action space
for i = 1:state_space       
    action_space{i} = actions;
end

number_of_action=5;

policy=zeros(state_space, number_of_action); % policy can be deterministic or stochastic, shown as follows:




% stochastic policy 

for i=1:state_space
    policy(i,:)=.2;          
end
% policy(3,2)=0; policy(3,4)=.4;
% policy(5,5)=0; policy(5,3)=.4;
policy(7,3)=1; policy(7,4)= 0; policy(7,2)= 0; policy(7,1)= 0; policy(7,5)= 0;
% policy(6,2)=0; policy(6,3) = 1; policy(6,4) = 0; policy(6,5) = 0; policy (6,1) = 0;




% Initialize the episode
episode_length = 1000;

state_history = zeros(episode_length, 2); 
reward_history = zeros(episode_length, 1);  

% Set the initial state
state_history(1, :) = agent_state;

for step = 1:episode_length
    action = stochastic_policy(state_history(step, :), action_space, policy, x_length, y_length);   
    % Calculate the new state and reward
    [new_state, reward] = next_state_and_reward(state_history(step, :), action, x_length, y_length, final_state, obstacle_state, reward_forbidden, reward_target, reward_step);
    % Update state and reward history
    state_history(step+1, :) = new_state;
    reward_history(step) = reward;
end

figure_plot(x_length, y_length, agent_state, final_state, obstacle_state, state_value, state, episode_length, state_history, policy, actions);

%%   useful function 
function [new_state, reward] = next_state_and_reward(state, action, x_length, y_length, target_state, obstacle_state, reward_forbidden, reward_target, reward_step)
    new_x = state(1) + action(1);
    new_y = state(2) + action(2);
    new_state = [new_x, new_y];

    % Check if the new state is out of bounds
    if new_x < 1 || new_x > x_length || new_y < 1 || new_y > y_length
        new_state = state;
        reward = reward_forbidden;
    elseif ismember(new_state, obstacle_state, 'rows')
        % If the new state is an obstacle
        reward = reward_forbidden;
    elseif isequal(new_state, target_state)
        % If the new state is the target state
        reward = reward_target;
    else
         % If the new state is a normal cell
        reward = reward_step;
    end
end

function action = stochastic_policy(state, action_space, policy, x_length, y_length)
    % Extract the action space and policy for a specific state
    state_1d = x_length * (state(2)-1) + state(1); 
    actions = action_space{state_1d};
    policy_i = policy(state_1d, :);

    % Ensure the sum of policy probabilities is 1
    assert(sum(policy_i) == 1, 'The sum of policy probabilities must be 1.');
    
    % Generate a random index based on policy probabilities
    action_index = randsrc(1, 1, [1:length(actions); policy_i]);
    
    % Select an action
    action = actions{action_index};
end
