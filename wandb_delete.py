import wandb

dry_run = False
api = wandb.Api()
proj_names = [ "plantain", "plantain_pose", "direct_bind" ]
for proj_name in proj_names:
    project = api.project(proj_name)
    for artifact_type in project.artifacts_types():
        for artifact_collection in artifact_type.collections():        
            for version in artifact_collection.versions():
                if artifact_type.type == 'model':
                    if len(version.aliases) > 0:
                        # print out the name of the one we are keeping
                        print(f'KEEPING {proj_name} {version.name}')
                    else:
                        print(f'DELETING {proj_name} {version.name}')
                        if not dry_run:
                            print('')
                            version.delete()
