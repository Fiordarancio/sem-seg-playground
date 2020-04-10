function F = choiceLoop(msg, choiceType)
%  CHOICELOOP - creates OK/CANCEL button to have a user interrupt a loop
% 
%  FS = CHOICELOOP creates a message box window and returns a structure FS 
%  that holds three functions, called FS.Ok, FS.Stop and FS.Clear. The
%  function Fs.Ok() will return true if the LEFT button has been pressed,
%  otherwise will return false when the RIGHT button is pressed. The string
%  displaying on those buttons can be chosen using choiceType. The function
%  FS.Stop() will return true if any of button has been clicked (or if the 
%  message box has been removed), so that a loop can be interrupted. The 
%  function FS.Clear() can be used to remove the message box, if a loop has
%  ended without user interruption.
%
%  FS = CHOICELOOP(MSG, CHOICETYPE) uses:
%       - the string MSG to display instead of the default message: 'Choose
%         OK to validate, else CANCEL.'
%       - the CHOICETYPE as option matching 'okcancel' (default) | 'yesno'
%         and  display different types of messages on the buttons
% 
% Adapted for MATLAB 2019b, inspired by STOPLOOP by Jos van der Gest
    nargoutchk(1,1);
    narginchk(0,2);
    if nargin
        if ~ischar(msg) && ~isstring(msg) && ~iscellstr(msg)
            error([mfilename ':InputString'],...
                'Input should be a string, or a cell array of strings.') ;
        else
            if ~ischar(choiceType) && ~isstring(choiceType) 
                error([mfilename ':InvalidOption'], ...
                    ['Expected type to be ''okcancel'' for OK-CANCEL options ' ...
                    'or ''yesno'' for YES-NO options.']);
            end
        end
    else
        % default message string
        msg = 'Choose OK to validate, else CANCEL.';
        % default type
        choiceType = 'okcancel';
    end
  
    % create a msgbox displaying the string
    H = msgbox(msg, 'STOPLOOP') ;
    % create the two anonymous functions
    F.Ok = @() okfun(H); % false if the right button is pressed, else true
    F.Stop = @() stopfun(H); % false if message box still exists
    F.Clear = @() clearfun(H) ; % delete message box
    
function r = stopfun(H)
    drawnow ;          % ensure that button presses are recorded
    r = ~ishandle(H) ; % false if message box still exists
end
function clearfun(H)
    % clear the message box if it still exists
    if ishandle(H)
        delete(H);
    end
end
function k = okfun(H)
        drawnow;
        k = true;
        end
end