close all;
clearvars -except old*;

filename = 'The Altogether - See the Day.wav';
[song,Fs] = audioread(filename);
SONG_MAX = length(song(:,1));
SONG_LEN = SONG_MAX/Fs;
SECS_PER_SEG = 2;
STARTS_PER_SEC = 64;

N=64;%N=88;
ZERO_FREQ=92.5;%ZERO_FREQ = 27.5;
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

nb_fs = @(t)(linearApproximation(nb_base, t*Fnb*(ZERO_FREQ/note_freq)+1));

note_max_ind = floor((length(nb_base)-2)/(ZERO_FREQ/note_freq));
note = @(i,t)...
    nb_fs(...
        (...%
            2^...
                (...
                    (i-1)/12 ...
                ).*...
            t.*...
            (...%
                2^...
                    (...
                        (i-1)/12 ...
                    ).*...
                t...
                <...
                note_max_ind/Fnb...
            )...
        )...
    ).*...
    (...%
        2^...
            (...
                (i-1)/12 ...
            ).*...
        t...
        <...
        note_max_ind/Fnb...
    )';

t = ((0:note_max_ind)/Fs)';
noteref = zeros(note_max_ind, N);
for i=1:N
    for j=1:length(t)
        noteref(j, i) = note(i, t(j));
    end
end

s = floor(Fs/STARTS_PER_SEC);
segment_len = s*floor(Fs/s)*SECS_PER_SEG;
n_segs = ceil(ceil(SONG_LEN)/SECS_PER_SEG);
appr2 = zeros(size(song(:,1)));
appr_num = 0;
lastlen = 0;
playvals = zeros(N, int64(ceil(SONG_MAX/s)));
xot = 0;
for cur_seg = 1:n_segs
    cprintf('*Red','Starting segment %d\n',cur_seg);
    [song,Fs] = audioread(filename, [(cur_seg-1)*segment_len+1, min(cur_seg*segment_len, SONG_MAX)]);
    song = song(:,1);
    song = song - appr2((cur_seg-1)*segment_len+1:(cur_seg-1)*segment_len+length(song));
    l = length(song);
    L = floor(l/s);
    if(lastlen ~= l)
        clear X beta beta_mul
        r = (zeros(note_max_ind*(N*L),1));
        c = (zeros(note_max_ind*(N*L),1));
        v = (zeros(note_max_ind*(N*L),1));
        nv = 0;
        lastpercent = -1;
        timerVal = -1;
        totalWork=0;
        doneWork=0;
        for j=int64(1:(N*L))
            n = floor((double(j)-1)/L)+1;
            i_max = ceil((note_max_ind+1)/2^(n/12));
            for i=int64(1:i_max)
                totalWork = totalWork + 1;
            end
        end
        tic;
        for j=int64(1:(N*L))
            rr=2^(-1/12);
            percent = floor(100*(doneWork/totalWork));
            if(percent > lastpercent)
                lastpercent = percent;
                if(lastpercent > 0)
                    timerVal = toc;
                end
                cprintf('*Black','(%d/%d) ',cur_seg, n_segs);
                fprintf("Calculating: %d%%", lastpercent);
                if(timerVal > 0)
                    fprintf("(ETA: %.2fs, elapsed: %.2fs)...",((totalWork-doneWork)/doneWork)*timerVal,timerVal);
                end
                fprintf("\n");
            end
            n = floor((double(j)-1)/L)+1;
            xo = s*mod(j-1,L);
            i_max = ceil((note_max_ind+1)/2^(n/12));
            for i=int64(1:i_max) 
                r(nv+i) = i+xo;
                c(nv+i) = j;
                v(nv+i) = noteref(i, n);%note(double(n), double(i-1)/Fnb);
                doneWork = doneWork+1;
            end
            nv = nv+i_max;
        end
        X = sparse(r(1:nv),c(1:nv),v(1:nv));
        clear r c v;
        Xs = X(1:length(song),:);
        beta_mul = (Xs'*Xs)\Xs';
        clear Xs;
        lastlen = l;
    end
    beta = beta_mul*song;
%     beta(beta>1)=1;
%     beta(beta<-1)=-1;
    playvals(:,int64(xot+1:xot+L)) = reshape(beta,L,N)';
%     for i=1:N
%         for j=1:L
%             playvals(i,j+xot) = beta(L*(i-1)+j);
%         end
%     end
%     for j=(1:length(beta))
%         playvals(...
%             int64(floor((double(j)-1)/L)+1),...
%             int64(xot+mod(j-1,L)+1)...
%         ) = beta(j);
%     end
    xot = xot + L;
    lastpercent = -1;
    if(appr_num+size(X,1) > size(appr2,1))
        appr2 = [appr2;zeros(appr_num+size(X,1)-size(appr2,1),1)];
    end
    appr2(appr_num+1:appr_num+size(X,1)) = appr2(appr_num+1:appr_num+size(X,1)) + gather(gpuArray(X)*gpuArray(beta));
    appr_num = appr_num+length(song);
end
clear X beta beta_mul
xot = xot-L;

for i = N:-1:1
    last_seg = ceil(ceil((note_max_ind+1)/2^((i-1)/12))/s);
    playvals_add = zeros(N, last_seg);
    playvals_add(i,1) = 1;
    playvals = [playvals_add playvals];
end

appr3 = zeros(int64((size(playvals,2)-1)*s+note_max_ind+1),1);

for i = 1:size(playvals, 2)
    appr3((i-1)*s+1:(i-1)*s+note_max_ind+1,:) = appr3((i-1)*s+1:(i-1)*s+note_max_ind+1,:) + noteref*playvals(:,i);
end

% 
% for i=1:size(playvals,1)
%     t = ((0:ceil((note_max_ind+1)/2^((i-1)/12)))/Fs)';
%     n = zeros(size(t));
%     for k=1:length(t)
%         n(k) = note(i-1,t(k));
%     end
%     tmp = conv(playvals(i,:)',n);
%     if(length(tmp) > length(appr3))
%         appr3 = appr3 + tmp(1:length(appr3));
%     else
%         appr3(1:length(tmp)) = appr3(1:length(tmp)) + tmp;
%     end
%     for j=1:size(playvals,2)
%         if((j-1)*s+length(n) > size(appr3,1))
%             appr3 = [appr3;zeros((j-1)*s+length(n)-size(appr3,1),1)];
%         end
%         appr3(int64((j-1)*s+1:(j-1)*s+length(n))) = appr3(int64((j-1)*s+1:(j-1)*s+length(n)))+playvals(i,j)*n;
%         
%         for k=0:ceil((note_max_ind+1)/2^((i-1)/12))
%             
%             if((j-1)*s+k+1<=length(appr3))
%                 appr3(int64((j-1)*s+k)+1) = appr3(int64((j-1)*s+k)+1) + playvals(i,j)*note(i-1, k/Fs);
%             end
%         end
%     end
% end
