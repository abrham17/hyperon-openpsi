!(bind! &demandspace (new-space))
(: Demand Type)
(: SkipDemand Type)
; (: demand (-> Symbol Number Demand))
(: skip (-> Symbol SkipDemand))

; (: addDemand (-> hyperon::space::DynSpace Symbol Number Demand))
(= (addDemand $space $name $value)
    (let* (
            ($demand (demand $name $value))
            ; ($duplicateDemandList  (match $space (demand $name $v) (demand $name $v)))
            (() (add-atom $space $demand))
        )
     ()
    )
    )

(= (fetch-demand $space $name)
    (match $space (demand $name $x) (demand $name $x))
)

(= (fetch-demand-val $space $name)
    (let ($x $y $z) (fetch-demand $space $name) $z)
)

(= (getAllDemands $space)
    (collapse (match $space (demand $x $_) (demand $x $_)))
)


(= (skipDemand $space $demand) 
    (let* (
        ($skip (collapse (match $space (skip $demand) (skip $demand))))
        ( $isNotFound (== $skip ()))
    )
    (if $isNotFound 
        (add-reduct $space (skip $demand)) 
        ()
    )
    )
)

(= (getEnabledDemands $space)
    (let* (
       ( $skippedDemands (collapse (match $space (, (demand $x $_) (skip $x)) (demand $x $_)) ))
    
       ( $allDemands (getAllDemands $space))
    
       ( $enabledDemands (collapse (subtraction  (superpose $allDemands)  (superpose $skippedDemands))))
     
    ) 
        $enabledDemands
    )
)

(= (setDemandValue $space $demandName $newValue)
    (let* (
        ((demand $demandName $oldValue ) (match $space (demand $demandName $oldValue) (demand $demandName $oldValue)))
        (() (remove-atom $space (demand $demandName $oldValue)))
        (() (add-atom $space (demand $demandName $newValue)))
    )
    ()
    )

)
(= (getDemandByName $space $name) (match $space (demand $name $x) (demand $name $x)))

; Fixed DemandDecay function to properly take space parameter
(= (DemandDecay $space $demand $value)
    (let* (
        ($oldValue  (match $space (demand $demand $oldValue) $oldValue))
        ($newValue (- $oldValue $value))
    )
    (if (>= $newValue 0)
        (setDemandValue $space $demand $newValue)
        (let $demandAtom (match $space (demand $demand $oldValue) (demand $demand $oldValue))
        (do
            (remove-atom $space $demandAtom)
            (remove-atom $space (skip $demand))
        )
    )
    )
))

; New function to apply decay only to skipped demands
(= (DecaySkippedDemands $space $decayValue)
    (let* (
        ($skippedDemandNames (collapse (match $space (skip $x) $x)))
        ($skippedDemands (collapse (match $space (demand $x $value) (demand $x $value))))
    )
    (if (== (size-atom $skippedDemands) 0 )
        (There is no skipped demand) 
        (DemandDecay $space (superpose $skippedDemandNames) $decayValue)
    ))
)
; New function to apply decay to all demands (both enabled and skipped)
(= (DecayAllDemands $space $decayValue)
    (let* (
        ($allDemands (getAllDemands $space))
        ($demandNames (collapse (match $space (demand $x $_) $x)))
    )
    (if (== (size-atom $allDemands) 0 )
        (No demands in the space)
        (DemandDecay $space (superpose $demandNames) $decayValue)
    ))
)

; New function to apply different decay rates to enabled vs skipped demands
(= (DecayDemandsWithDifferentRates $space $enabledDecayRate $skippedDecayRate)
    (let* (
        ($enabledDemandNames (getEnabledDemands $space))
        ($enabledDemands (collapse (unify $enabledDemandNames (demand $x $_) $x ())))
        ($skippedDemandNames (collapse (match $space (skip $x) $x)))
        ($_ (DecaySkippedDemands $space $skippedDecayRate))
    )
    (if (== (size-atom $enabledDemands) 0)
        (There is no skipped demand)
        (DemandDecay $space (superpose $enabledDemands) $enabledDecayRate)
        
    ))
)
!(addDemand &demandspace energy 0.6)
!(addDemand &demandspace affiliation 0.5)
!(addDemand &demandspace competence 0.4)
!(skipDemand &demandspace competence)
!(getEnabledDemands &demandspace)
!(DecaySkippedDemands &demandspace 0.1)
!(DecayAllDemands &demandspace 0.05)
!(DecayDemandsWithDifferentRates &demandspace 0.05 0.5)

!(get-atoms &demandspace)