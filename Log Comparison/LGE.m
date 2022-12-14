function P = LGE(a, t, x0)
P = a(1)./(1+((a(1)-x0)./x0).*exp(-1*a(2).*(t-3)));
end