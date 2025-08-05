;; An implementation of pseudo random generators for meTTalog
!(register-module! ../../hyperon-openpsi)
!(import! &self hyperon-openpsi:mettalog:math-functions)
!(import! &self hyperon-openpsi:mettalog:general-helpers)

!(bind! &rand-gen (new-space))
!(add-atom &rand-gen 42)
;; An implementation of the linear congruential generator. This implementation takes seed values to generate a pseudo-random number
;; which is going to be used in the mettalog implementation.
(: linear-congruential-generator (-> Number Number Number Number Number))
(= (linear-congruential-generator $seed $a $c $m)
   (% (+ (* $seed $a) $c) $m))

(: rand-float (-> Number))
(= (rand-float)
   (let* (
          ($seed (get-atoms &rand-gen))
          ($_ (remove-atom &rand-gen $seed))
          ($next (linear-congruential-generator $seed 1664525 1013904223 42949672967296))
          ($_ (add-atom &rand-gen $next))
          )
            (/ $next 42949672967296) 
))
(: rand-float-range (-> Number Number Number))
(= (rand-float-range $min $max)
   (if (>= $min $max)
        (Error "The first argument should be less than the second argument")
        (let* (
               ($rand_res (rand-float))
               ($adder (* (- $max $min) $rand_res))
               ($res (+ $min $adder)
               
               )
            )
            $res
        )
    )
)
;;An implementation of the normal function based on the box-muller transform

(= (pseudo-normal' $mu $sigma)
    (let* (
       ($u1 (rand-float))
       ($u2 (rand-float))
       
       ($r (sqrt-math (* -2.0 (ln $u1))))
       ($theta (* (* 2.0 3.14159265) $u2))
       ($z (* $r (cos-math $theta)))

    )
        (+ $mu (* $sigma $z)) 
    )
)
(= (pseudo-normal)
    (pseudo-normal' 0 1)
)
;; !(>= 45 5)
;; !(<= 45 5)
;; !(rand-float-range 10 15)
;; !(rand-float)
;; !(rand-float)
;; !(rand-float)
;; !(pseudo-normal)
;;!(log-math 2 13)
;; !(pow-math 2 3.3)
;;!(% 15 2)
;;!(<< 15 3)