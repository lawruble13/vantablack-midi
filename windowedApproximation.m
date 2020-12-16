close all;
clearvars -except old*;

filename = 'br.wav';
[song,Fs] = audioread(filename);
SONG_MUL = 0.8/max(song(:));
SONG_MAX = length(song(:,1));
SONG_LEN = SONG_MAX/Fs;
plt = axes;
hold on;
fs_ln = plot((0:SONG_MAX-1)/Fs, SONG_MUL*sum(song,2)/size(song,2));

SECS_PER_SEG = 3;
STARTS_PER_SEC = 96;

WINDOW_LEN = 1.5;
WINDOW_OVERLAP = 2/3;

N=88;
ZERO_FREQ = 27.5;
nb_filename = 'bass.wav';

[nb_base, Fnb] = audioread(nb_filename);
nb_base = nb_base(:,1);
nb_base = nb_base(1:2*floor(length(nb_base)/2));
Ft_nb = fft(nb_base);
P = abs(Ft_nb/length(nb_base));
P = P(1:length(nb_base)/2+1);
P(2:end-1) = 2*P(2:end-1);
f = Fnb*(0:(length(nb_base)/2))/length(nb_base);
[~, f_ind] = max(P);
note_freq = f(f_ind);
clear Ft_nb P f;

note_max_ind = floor((length(nb_base)-2)/(ZERO_FREQ/note_freq));
while note_max_ind > WINDOW_LEN*WINDOW_OVERLAP*Fs
    fprintf('Error: desired zero frequency exceeds window overlap\n')
    ZERO_FREQ = ZERO_FREQ * 2^(1/12);
    N = N - 1;
    note_max_ind = floor((length(nb_base)-2)/(ZERO_FREQ/note_freq));
end
fprintf('Beginning calculation with zero frequency %.2f, and N=%d\n', ZERO_FREQ, N)

nb_fs = @(t)(linearApproximation(nb_base, t*Fnb*(ZERO_FREQ/note_freq)+1));
note = @(i,t)...
    nb_fs(...
        2^((i-1)/12).*t.*...
        (2^((i-1)/12).*t < note_max_ind/Fnb)...
    ).*...
    (...
        2^((i-1)/12).*t < note_max_ind/Fnb...
    )';

t = ((0:note_max_ind)/Fs)';
noteref = zeros(note_max_ind, N);
for i=1:N
    for j=1:length(t)
        noteref(j, i) = note(i, t(j));
    end
end

window_width = floor(WINDOW_LEN*Fs);
window_movement = floor(WINDOW_LEN*(1-WINDOW_OVERLAP)*Fs);
window_left = 1;
window_num = ceil((SONG_MAX-window_width)/window_movement)+1;

playtimes = floor((1:(Fs/STARTS_PER_SEC):SONG_MAX));
playvals = zeros(N, length(playtimes));
appr = zeros(size(song,1)+note_max_ind, 1);
appr_t = (0:length(appr)-1)/Fs;
appr_ln = plot(appr_t, appr);
appr_ln.YDataSource = 'appr';

for window_n = 1:window_num
    fprintf('Starting window %d/%d\n',window_n,window_num);
    window_right = min(window_left-1+window_width, SONG_MAX);
    [song, Fs] = audioread(filename, [window_left, window_right]);
    song = SONG_MUL * (sum(song,2)/size(song,2));
    song = song - appr(window_left:window_right);
    new_window_offsets = playtimes(window_left <= playtimes & playtimes < window_right) - (window_left);
    window_playinds = find(window_left <= playtimes & playtimes < window_right);
    L = length(window_playinds);
    if ~exist('window_offsets', 'var') || ~isequal(new_window_offsets, window_offsets)
        clear beta_mul X
        window_offsets = new_window_offsets;
        fprintf('Recalculating matrix: ');
        totalItems = 0;
        addedItems = 0;
        for n=1:N
            totalItems = totalItems + L*ceil((note_max_ind+1)/2^(n/12));
        end
        row = zeros(totalItems, 1);
        col = zeros(totalItems, 1);
        val = zeros(totalItems, 1);
        tStart = tic; tic;
        status_len = fprintf('[%s] Time Elapsed: 0s, ETA:', repmat('░',1,20));
        for n = 1:N
            for i=1:L
                if toc > 1
                    nblocks = round(20 * addedItems/totalItems);
                    fprintf(repmat('\b',1,status_len));
                    tElapsed = toc(tStart);
                    status_len = fprintf('[%s] Time Elapsed: %.2fs, ETA: %.2fs',...
                        pad(repmat('█',1,nblocks),20,'right','░'),...
                        tElapsed,...
                        ((totalItems-addedItems)/addedItems)*tElapsed);
                    tic;
                end
                t_max = ceil((note_max_ind+1)/2^(n/12));
                for t = 1:t_max
                    row(addedItems+t) = window_offsets(i) + t;
                    col(addedItems+t) = (n-1)*L + i;
                    val(addedItems+t) = noteref(t, n);
                end
                addedItems = addedItems + t_max;
            end
        end
        X = sparse(row, col, val);
        clear row col val;
        estimated_size = 16*length(song)*L*N;
        [~, sys] = memory;
        if estimated_size < sys.PhysicalMemory.Total
            Xs = full(X(1:length(song),:));
            beta_mul = (Xs'*Xs)\Xs';
            clear Xs;
        end
        fprintf(repmat('\b',1,status_len));
        tElapsed = toc(tStart);
        fprintf('Done, Time Taken: %.2fs\n', tElapsed);
    end
    
    fprintf('Calculating playvalues: ');
    tic;
    if exist('beta_mul', 'var')
        beta = beta_mul*song;
        clear appr_t;
        fprintf('Done, Time Taken: %.2fs\n', toc);
    else
        p = gcp();
        future = parfeval(p, @lsqminnorm, 1, X(1:length(song),:), song);
        spinner = ['|','/','-','\'];
        i = 0;
        status_len = 0;
        while ~strcmp(future.State, 'finished')
            pause(0.3);
            fprintf(repmat('\b',1,status_len));
            status_len = fprintf('%s Time Elapsed: %.2fs', spinner(i+1), toc);
            i = mod(i+1,4);
        end
        fprintf(repmat('\b',1,status_len));
        if ~isempty(future.Error)
            fprintf('%s\n', future.Error.message);
            break;
        else
            beta = fetchOutputs(future);
            fprintf('Done, Time Taken: %.2fs\n', toc);
        end
    end
    
    fprintf('Determining playvalues and approximation: ');
    tic;
    playvals(:,window_playinds) = playvals(:,window_playinds) + reshape(beta, L, N)';
    
    appr(window_left:(window_left-1+size(X,1))) = appr(window_left:(window_left-1+size(X,1))) + X*beta;
    refreshdata(appr_ln);
    fprintf('Done, Time Taken: %.2fs\n', toc);
    window_left = window_left + window_movement;
end

rt = 0;
playvals = [zeros(N, N) playvals];
playtimes = [zeros(1, N) playtimes];
for i = 1:N
	playvals(i, i) = 1;
    playtimes(i) = rt+1;
    rt = rt + ceil((note_max_ind+1)/2^((i-1)/12));
end
playtimes(N+1:end) = playtimes(N+1:end)+rt;

appr2 = zeros(max(playtimes)+note_max_ind, 1);
for i = 1:size(playtimes,2)
    ts = playtimes(i);
    appr2(ts:ts+note_max_ind) = appr2(ts:ts+note_max_ind) + noteref*playvals(:,i);
end