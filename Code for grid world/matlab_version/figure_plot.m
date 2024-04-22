% by the Intelligent Unmanned Systems Laboratory, Westlake University, 2024

function figure_plot(x_length, y_length, agent_state,final_state, obstacle_state, state_value, state_number, episode_length,state_update_2d,policy, action)
    %% Inverse y coordinate
 
    
    xa_used = agent_state(:, 1) + 0.5;
    ya_used = y_length+1-agent_state(:, 2) + 0.5;
    
    
    state_space=x_length*y_length;
    
    
    xf = final_state(:, 1);
    yf = y_length+1-final_state(:, 2); 
    
    
    
    xo = obstacle_state(:, 1);
    yo = y_length+1-obstacle_state(:, 2);
    
    
    
    xs = state_update_2d(:, 1); 
    ys = state_update_2d(:, 2);
    
    state_update = (ys-1) * x_length + xs; 
                                                        
    
    
    %%
    
    greenColor=[0.4660 0.6740 0.1880]*0.8;
    
    
    
    % Initialize the figure
    figure();
    
    
    % Add labels on the axes
    addAxisLabels(x_length, y_length);
    
    % Draw the grid, state values, and policy arrows
    r = drawGridStateValuesAndPolicy(x_length, y_length, state_number, state_value, policy, greenColor, action);
    
    % Color the obstacles and the final state
    colorObstacles(xo, yo, r);
    colorFinalState(xf, yf, r);
    
    % Draw the agent
    agent = plot(xa_used, ya_used, '*', 'markersize', 15, 'linewidth', 2, 'color', 'b');  
    hold on;
    
    axis equal
    axis off
    exportgraphics(gca,'policy_offline_Q_learning.pdf') 
    
    
    
    % Initialize the figure
    figure();
    
    % Add labels on the axes
    addAxisLabels(x_length, y_length);
    
    % Draw the grid and add state values
    r = drawGridAndStateValues(x_length, y_length, state_value);
    
    % Color the obstacles and the final state
    colorObstacles(xo, yo, r);
    colorFinalState(xf, yf, r);
    
    % Compute the de-normalized states
    for i = 1:state_space
        state_two_dimension_new(i, :) = de_normalized_state(state_number(i), x_length, y_length);
    end
    
    
    % Draw the agent
    agent = plot(xa_used, ya_used, '*', 'markersize', 15, 'linewidth', 2, 'color', 'b');
    hold on;
    
    
    % Set axis properties and export the figure
    axis equal;
    axis off;
    exportgraphics(gca, 'trajectory_Bellman_Equation.pdf');
    
    
    
    
    
    
    % Initialize the figure
    figure();
    
    % Add labels on the axes
    addAxisLabels(x_length, y_length);
    
    % Draw the grid and add state values
    r= drawGridAndStateValues(x_length, y_length, state_value);
    
    % Draw state transitions
    for i=1:state_space
        state_two_dimension_new(i,:)=de_normalized_state(state_number(i),x_length,y_length);
    end
    drawStateTransitions(state_space, state_update, state_two_dimension_new, episode_length);
    
    % Color the obstacles and the final state
    
    
    colorObstacles(xo, yo, r);
    colorFinalState(xf, yf, r);
    
    
    % Draw the agent
    agent = plot(xa_used, ya_used, '*', 'markersize', 15, 'linewidth', 2, 'color', 'b');
    hold on;
    
    
    % Set axis properties and export the figure
    axis equal;
    axis off;
    exportgraphics(gca, 'trajectory_Q_learning.pdf');
    
    
    
    
    % Initialize the figure
    figure();
    
    % Add labels on the axes
    addAxisLabels(x_length, y_length);
    
    % Draw the grid and add state values
    r = drawGridAndStateValues(x_length, y_length, state_value);
    
    % Color the obstacles and the final state
    colorObstacles(xo, yo, r);
    colorFinalState(xf, yf, r);
    
    % Compute the de-normalized states
    for i = 1:state_space
        state_two_dimension_new(i, :) = de_normalized_state(state_number(i), x_length, y_length);
    end
    
    % Draw transitions between states
    for i = 1:episode_length - 1
        line([state_two_dimension_new(state_update(i), 1) + 0.5, state_two_dimension_new(state_update(i + 1), 1) + 0.5], ...
             [state_two_dimension_new(state_update(i), 2) + 0.5, state_two_dimension_new(state_update(i + 1), 2) + 0.5], ...
             'Color', 'black', 'LineStyle', '--');
        hold on;
    end
    
    % Draw the agent
    agent = plot(xa_used, ya_used, '*', 'markersize', 15, 'linewidth', 2, 'color', 'b');
    hold on;
    
    
    % Set axis properties and export the figure
    axis equal;
    axis off;
    exportgraphics(gca, 'trajectory_Bellman_Equation.pdf');
    
    % Function definitions would be the same as provided previously


end



function o=de_normalized_state(each_state,x_length,y_length)

         o=[mod(each_state-1,x_length),y_length-1-fix((each_state-1)/(x_length))]+[1,1];
end




function addAxisLabels(x_length, y_length)
    for i = 1:x_length
        text(i + 0.5, y_length + 1.1, num2str(i));
    end
    for j = y_length:-1:1
        text(0.9, j + 0.5, num2str(y_length - j + 1));
    end
end

function r= drawGridStateValuesAndPolicy(x_length, y_length, state_number, state_value, policy, greenColor, action)
    ind = 0;
    ratio = 0.5; % adjust the length of arrow
    state_coordinate = zeros(x_length * y_length, 2); % Initialize state_coordinate
    for j = y_length:-1:1       
        for i = 1:x_length      
            r(i, j) = rectangle('Position', [i j 1 1]);
            ind = ind + 1;
            state_coordinate(state_number(ind), :) = [i, j];
            text(i + 0.4, j + 0.5, ['s', num2str(ind)]);
            hold on;
            
            % Calculate bias
            i_bias(ind) = state_coordinate(state_number(ind), 1) + 0.5;
            j_bias(ind) = state_coordinate(state_number(ind), 2) + 0.5;
            
            % Draw policy arrows or state markers
            for kk = 1:size(policy, 2)
                if policy(state_number(ind), kk) ~= 0
                    kk_new = policy(state_number(ind), kk) / 2 + 0.5;
                    drawPolicyArrow(kk, ind, i_bias, j_bias, kk_new, ratio, greenColor, action);                
                end
            end
        end
    end
end


function drawPolicyArrow(kk, ind, i_bias, j_bias, kk_new, ratio, greenColor, action)
    % Obtain the action vector
    action = action{kk};

    % For the non-moving action, draw a circle to represent the stay state
    if action(1) == 0 && action(2) == 0  % Assuming the fifth action is to stay
        plot(i_bias(ind), j_bias(ind), 'o', 'MarkerSize', 8, 'linewidth', 2, 'color', greenColor);
        return;
    else
        % Draw an arrow to represent the moving action; note that '-' used when drawing the y-axis arrow ensures consistency with the inverse y-coordinate handling.
        arrow = annotation('arrow', 'Position', [i_bias(ind), j_bias(ind), ratio * kk_new * action(1), - ratio * kk_new * action(2)], 'LineStyle', '-', 'Color', greenColor, 'LineWidth', 2);
        arrow.Parent = gca;
    end
end


% Function to draw the grid and add state values
function r = drawGridAndStateValues(x_length, y_length, state_value)
    ind = 0;
    for j = y_length:-1:1       
        for i = 1:x_length       
            r(i, j) = rectangle('Position', [i j 1 1]);
            ind = ind + 1;
            text(i + 0.4, j + 0.5, num2str(round(state_value(ind), 2)));
            hold on;           
        end
    end
end

% Function to color the obstacles
function colorObstacles(xo, yo, r)
    for i = 1:length(xo)
        r(xo(i), yo(i)).FaceColor = [0.9290 0.6940 0.1250];
    end
end

% Function to color the final state
function colorFinalState(xf, yf, r)
    r(xf, yf).FaceColor = [0.3010 0.7450 0.9330];
end

% Function to draw state transitions
function drawStateTransitions(state_space, state_update, state_two_dimension_new, episode_length)
    for i = 1:episode_length - 1
        if state_two_dimension_new(state_update(i), 2) ~= state_two_dimension_new(state_update(i + 1), 2)       
            line([state_two_dimension_new(state_update(i), 1) + 0.5, state_two_dimension_new(state_update(i), 1) + 0.5 + 0.03 * randn(1), state_two_dimension_new(state_update(i + 1), 1) + 0.5 + 0.03 * randn(1), state_two_dimension_new(state_update(i + 1), 1) + 0.5], ...
                 [state_two_dimension_new(state_update(i), 2) + 0.5, state_two_dimension_new(state_update(i), 2) + 0.25 + 0.03 * randn(1), state_two_dimension_new(state_update(i + 1), 2) + 0.75 + 0.03 * randn(1), state_two_dimension_new(state_update(i + 1), 2) + 0.5], ...
                 'Color', 'green');
        elseif state_two_dimension_new(state_update(i), 1) ~= state_two_dimension_new(state_update(i + 1), 1)       
            line([state_two_dimension_new(state_update(i), 1) + 0.5, state_two_dimension_new(state_update(i), 1) + 0.25 + 0.03 * randn(1), state_two_dimension_new(state_update(i + 1), 1) + 0.75 + 0.03 * randn(1), state_two_dimension_new(state_update(i + 1), 1) + 0.5], ...
                 [state_two_dimension_new(state_update(i), 2) + 0.5, state_two_dimension_new(state_update(i), 2) + 0.5 + 0.03 * randn(1), state_two_dimension_new(state_update(i + 1), 2) + 0.5 + 0.03 * randn(1), state_two_dimension_new(state_update(i + 1), 2) + 0.5], ...
                 'Color', 'green');
        end
        hold on;
    end
end


