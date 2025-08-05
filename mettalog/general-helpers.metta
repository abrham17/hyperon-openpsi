(: >= (-> Number Number Bool))
(= (>= $fir $sec) (if (or (> $fir $sec) (== $fir $sec)) True False))
(: <= (-> Number Number Bool))
(= (<= $fir $sec) (if (or (< $fir $sec) (== $fir $sec)) True False))