function [lin_app] = linearApproximation(f_in,i_app)
i_l = fix(i_app);
i_u = i_l+1;
f_l = f_in(i_l);
f_u = f_in(i_u);

lin_app = f_l + (f_u-f_l)*(i_app-i_l);
end

