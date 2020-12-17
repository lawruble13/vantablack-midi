close all;
clearvars -except old*;

%% User defined variables
% Defines variables: song_filename, note_filename, segment_len, patch_width,
% zero_freq, num_notes, play_rate
% Clears variables: none

% The filename of the song to approximate, in the folder ./Song\ Inputs
song_filename = 'The Altogether - See the Day.wav';
% The filename of the note to use, in the folder ./Note\ Inputs
note_filename = 'bass.wav';
% The length of the sequential segments, in seconds
segment_len = 1;
% The width of the patch segments, in seconds (<= segment_len)
patch_width = 4/3;
% The desired frequency of the lowest note, in Hz (27.5 = A0)
zero_freq = 27.5;
% The desired number of different notes
num_notes = 88;
% The rate to play notes at, in Hz
play_rate = 64;

%% Load the requested song
% Defines variables: Fs, song
% Clears variables: none

% Read the target song audio file
[song, Fs] = audioread(append('Song Inputs/', song_filename));
% Convert the song to mono audio
song = sum(song, 2)/size(song, 2);

%% Determine the playtime offsets for segments and patches
% Defines variables: actual_samples_per_seg, seg_offsets,
% actual_samples_before_patch, patch_offsets
% Clears variables: none
desired_samples_per_play = Fs / play_rate;
actual_samples_per_play = floor(desired_samples_per_play);

desired_samples_per_seg = Fs * segment_len;
actual_plays_per_seg = ceil(desired_samples_per_seg / actual_samples_per_play);
actual_samples_per_seg = actual_samples_per_play * actual_plays_per_seg;

seg_offsets = (0:actual_plays_per_seg-1)*actual_samples_per_play;

desired_patch_offset = (segment_len - patch_width/2)*Fs;
actual_plays_before_patch = floor(desired_patch_offset / actual_samples_per_play);
actual_samples_before_patch = actual_plays_before_patch * actual_samples_per_play;
desired_plays_per_patch = 2*(actual_samples_per_seg-actual_plays_before_patch);
actual_plays_per_patch = min(desired_plays_per_patch, actual_plays_per_seg);
actual_samples_per_patch = actual_plays_per_patch * actual_samples_per_play;

patch_offsets = (0:actual_plays_per_patch-1)*actual_samples_per_play;
clear desired*

%% Adjust the song to be a whole number of sequential segments
n_seq_seg = ceil(length(song)/actual_samples_per_seg);
song(end+1:n_seq_seg*actual_samples_per_seg) = 0;
song_max = length(song);

%% Process the requested note
% Defines variables: note, Fsn, note_freq
% Clears variables: none

% Read the requested note audio file
[note, Fsn] = audioread(append('Note Inputs/', note_filename));
% Convert the note to mono audio
note = sum(note, 2)/size(note, 2);
% FFT used to obtain frequency of note prior to shifting
% Make the length of the note even for FFT
if mod(length(note), 2) == 1
    note(end+1) = 0;
end
% Perform FFT, realize, then make one-sided
note_fft = fft(note);
note_fd_pow = abs(note_fft/length(note));
note_fd_pow = note_fd_pow(1:length(note)/2+1);
note_fd_pow(2:end-1) = 2*note_fd_pow(2:end-1);
% Create array of frequency values
% The power and frequency arrays have the same index
f = Fsn*(0:(length(note)/2))/length(note);
% Find the frequency with max power
[~, f_ind] = max(note_fd_pow);
note_freq = f(f_ind);
clear note_fft note_fd_pow f f_ind;

%% Check the requested frequency
% Defines variables: none
% Clears variables: none

% Determine the minimum frequency so the whole note fits within the patch
max_len_ratio = (actual_samples_per_patch/2)/length(note);
max_semitones_down = log(max_len_ratio) / log(2^(1/12));
min_freq = 2^(-max_semitones_down/12) * note_freq;
% If they requested a lower frequency than is possible, let them know, and
% set the zero frequency to the lowest frequency possible that is in tune
% with the requested frequency, and lower the number of notes
% commensurately
if min_freq > zero_freq
    fprintf("Error: requested zero frequency %.2f is less than the minimum frequency %.2f\n", zero_freq, min_freq);
    semitones_different = 12*log(min_freq/zero_freq)/log(2);
    zero_freq = zero_freq * 2^(ceil(semitones_different)/12);
    num_notes = num_notes - ceil(semitones_different);
    fprintf("\tUsing %.2f for zero frequency, and %d notes.\n", zero_freq, num_notes);
    clear semitones_different
end
clear max_len_ratio max_semitones_down min_freq

%% Create the note reference
% Defines variables: note_max_ind, noteref
% Clears variables: note, note_freq, Fsn, zero_freq


% Determine the maximum meaningful index for any note
note_max_ind = floor((length(note)-2)/(zero_freq/note_freq));
% Use linear approximation to resample the note at arbitrary times, and to
% pitch shift it to the determined zero frequency
note_linapp = @(t)(linearApproximation(note, t*Fsn*(zero_freq/note_freq)+1));
% Calculate equivalent time for the note pitch-shifted up by i-1 semitones
note_eqtime = @(i,t)(2^((i-1)/12).*t);
% Check a time is valid for a given pitch-shift
note_valtime = @(i,t)(note_eqtime(i,t)<note_max_ind/Fsn);
% Use this approximation to pitch-shift the note by i-1 semitones up from 
% the zero frequency
note_ps = @(i,t)(note_linapp(note_eqtime(i,t).*note_valtime(i,t)).*note_valtime(i,t));
t = ((0:note_max_ind)/Fs)';
noteref = zeros(note_max_ind, num_notes);
for i=1:num_notes
    for j = 1:length(t)
        noteref(j, i) = note_ps(i, t(j));
    end
end
clear note note_linapp note_eqtime note_valtime note_ps t i j note_freq Fsn zero_freq

%% Calculate the OLS matrix for the segments
% Calculate the number of items per play
items_per_play = 0;
for n=1:num_notes
    items_per_play = items_per_play + ceil((note_max_ind+1)/2^(n/12));
end
% Calculate the total number of items per segment
total_items = length(seg_offsets)*items_per_play;
% Calculate the sparse matrix for the segments
row = zeros(total_items, 1);
col = zeros(total_items, 1);
val = zeros(total_items, 1);
added_items = 0;
for n=1:num_notes
    for i=1:length(seg_offsets)
        t_max = ceil((note_max_ind+1)/2^(n/12));
        for t = 1:t_max
            row(added_items+t) = seg_offsets(i) + t;
            col(added_items+t) = (n-1)*length(seg_offsets) + i;
            val(added_items+t) = noteref(t, n);
        end
        added_items = added_items + t_max;
    end
end
X = sparse(row, col, val);
clear row col val;
Xs = full(X(1:actual_samples_per_seg,:));
beta_mul = ((Xs'*Xs)\Xs');
clear Xs;


%% Calculate for each sequential segment
song_segs = reshape(song, [], n_seq_seg);
playvals = beta_mul*song_segs;
clear beta_mul song_segs
%% Calculate approximation before patches
approximation = zeros(length(song)+note_max_ind, 1);
for i=1:n_seq_seg
    seg_left = actual_samples_per_seg*(i-1)+1;
    seg_appr = X*playvals(:,i);
    approximation(seg_left:seg_left-1+length(seg_appr)) = approximation(seg_left:seg_left-1+length(seg_appr)) + seg_appr;
    clear seg_appr;
end

%% Calculate the OLS matrix for the patches
% Calculate the number of items per play
items_per_play = 0;
for n=1:num_notes
    items_per_play = items_per_play + ceil((note_max_ind+1)/2^(n/12));
end
% Calculate the total number of items per segment
total_items = length(patch_offsets)*items_per_play;
% Calculate the sparse matrix for the segments
row = zeros(total_items, 1);
col = zeros(total_items, 1);
val = zeros(total_items, 1);
added_items = 0;
for n=1:num_notes
    for i=1:length(patch_offsets)
        t_max = ceil((note_max_ind+1)/2^(n/12));
        if patch_offsets(i) + note_max_ind < actual_samples_per_patch
            for t = 1:t_max
                row(added_items+t) = patch_offsets(i) + t;
                col(added_items+t) = (n-1)*length(patch_offsets) + i;
                val(added_items+t) = noteref(t, n);
            end
            added_items = added_items + t_max;
        end
    end
end
row = row(1:added_items);
col = col(1:added_items);
val = val(1:added_items);

X = sparse(row, col, val);
clear row col val;
Xs = full(X);
Xs(end+1:actual_samples_per_patch,:) = 0;

beta_mul = ((Xs'*Xs)\Xs');
clear Xs;

%%
error = song-approximation(1:length(song));
error = error(actual_samples_before_patch+1:end);
error = error(1:actual_samples_per_patch*(n_seq_seg-1));
error_segs = reshape(error, actual_samples_per_patch, n_seq_seg-1);
clear error;
playvals=beta_mul*error_segs;

clear beta_mul;

%% Calculate approximation after patches
approximation_patch = zeros(length(song)+note_max_ind, 1);
for i=1:n_seq_seg-1
    patch_left = actual_samples_per_seg*(i-1)+actual_samples_before_patch + 1;
    patch_right = patch_left - 1 + size(X, 1);
    patch_appr = X*playvals(:,i);
    approximation_patch(patch_left:patch_right) = approximation_patch(patch_left:patch_right) + patch_appr;
    clear seg_appr;
end

%% Combine pre-patch appproximation and patches
approximation = approximation + approximation_patch;
