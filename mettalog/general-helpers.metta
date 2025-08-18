(: >= (-> Number Number Bool))
(= (>= $fir $sec) (if (or (> $fir $sec) (== $fir $sec)) True False))
(: <= (-> Number Number Bool))
(= (<= $fir $sec) (if (or (< $fir $sec) (== $fir $sec)) True False))

(= (Member-check $x $list)
    (let* (($superpose_x (superpose $list)))
            (let $netVal (intersection $x $superpose_x) (not (== $netVal ())))
))

(= (exp $x)
    (pow-math 2.71828 $x)
)

(= (fold $f $acc $tuple)
   (if (== $tuple ())
       $acc
       (let $next (car-atom $tuple) (fold $f ($f $next $acc) (cdr-atom $tuple)))
   )
)
