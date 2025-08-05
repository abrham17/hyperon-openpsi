
(: series_summation (-> Number Number Number Number))
(= (series_summation $y $iter $accum)
    (if (< $iter 0)
        (* 2 $accum)
        (let* (
            ($l_val (/ 1 (+ (* 2 $iter) 1)))
            ($r_val  (pow-math $y (+ (* 2 $iter) 1)))
            ($term (* $l_val $r_val))
            ($accum' (+ $accum $term))
        )
            (series_summation $y (- $iter 1) $accum')
        )
    )
)

;; This function computes the x multiplicand which is independent of the logarithmic calculation.
(: x_multiplicand (-> Number Number))
(= (x_multiplicand $x) (/ (- $x 1) (+ $x 1)))

(= (ln $x)
    (if (< $x 0)
        (Error "The x value is an error")
        (let* (
            ($y (x_multiplicand $x))
            ($result (series_summation $y 100 0))
        )
            $result
        )
    )
)
;; based on the laws of logarithms
(= (log $x $base)
    (/ (ln $x) (ln $base))
)

!(ln 7.6)